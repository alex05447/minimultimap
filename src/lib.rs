//! # minimultimap
//!
//! A simple minimalistic multimap wrapper around a [`HashMap`](std::collections::hash_map::HashMap),
//! optimized for the usual case of having one value per key with no overhead
//! (i.e. a [`Vec`] is not allocated to hold a single value),
//! but which does support multiple (potentially non-unique) values by storing them in a [`Vec`] when necessary.

use {
    miniunchecked::{unreachable_dbg, OptionExt, ResultExt},
    std::{
        borrow::Borrow,
        collections::{
            hash_map::{
                Drain, Entry as HashMapEntry, IntoIter, IntoKeys, IntoValues, Iter, IterMut, Keys,
                OccupiedEntry as HashMapOccupiedEntry, RandomState,
                VacantEntry as HashMapVacantEntry, Values, ValuesMut,
            },
            HashMap, TryReserveError,
        },
        error::Error,
        fmt::{self, Debug, Display, Formatter},
        hash::{BuildHasher, Hash},
        iter::{FusedIterator, Iterator},
        num::NonZeroUsize,
        ops::{Deref, DerefMut, Index},
        ptr::NonNull,
        slice::{self, Iter as SliceIter, IterMut as SliceIterMut},
    },
};

/// One or multiple values in the [`MultiMap`] entry associated with a given key.
///
/// Derefs to a (non-empty) slice.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum EntryValues<V> {
    /// A single value associated with a key.
    One(V),
    /// Two or more values associated with a key.
    Multiple(Vec<V>),
}

/// Error kind returned when trying to insert a new value into the [`MultiMap`] entry.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum InsertErrorKind {
    /// Insert index is out of bounds.
    /// Contains the (non-zero) number of existing values.
    IndexOutOfBounds(NonZeroUsize),
    /// Inserted value is not unique.
    ValueNotUnique,
}

impl Display for InsertErrorKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            InsertErrorKind::IndexOutOfBounds(size) => {
                write!(f, "insert index is out of bounds (size is {size})")
            }
            InsertErrorKind::ValueNotUnique => write!(f, "inserted value is not unique"),
        }
    }
}

/// Error returned when inserting a new value into the [`MultiMap`] entry.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct InsertError<V> {
    /// Non-inserted value, returned back.
    pub value: V,
    /// Error kind.
    pub error: InsertErrorKind,
}

impl<V> InsertError<V> {
    fn index_out_of_bounds(value: V, len: NonZeroUsize) -> Self {
        Self {
            value,
            error: InsertErrorKind::IndexOutOfBounds(len),
        }
    }

    fn value_non_unique(value: V) -> Self {
        Self {
            value,
            error: InsertErrorKind::ValueNotUnique,
        }
    }
}

impl<V: Debug> Display for InsertError<V> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "value {:?} not inserted: {}", self.value, self.error)
    }
}

impl<V: Debug> Error for InsertError<V> {}

/// Result returned when removing a value from the (non-empty) [`EntryValues`].
#[derive(Clone, PartialEq, Eq, Debug)]
struct EntryValuesRemoveResult<V> {
    /// Removed value.
    removed: V,
    /// If the removed `value` was not last, the rest of the remaining values.
    remaining: Option<EntryValues<V>>,
}

impl<V> EntryValuesRemoveResult<V> {
    fn last(removed: V) -> Self {
        Self {
            removed,
            remaining: None,
        }
    }

    fn non_last(removed: V, remaining: EntryValues<V>) -> Self {
        Self {
            removed,
            remaining: Some(remaining),
        }
    }
}

impl<V> EntryValues<V> {
    /// Tries to insert the (non-unique) `value` at `index` into these [`EntryValues`].
    /// Returns `None` on success, else returns the `value` back if `index` is out of bounds.
    fn insert(&mut self, index: usize, value: V) -> Option<V> {
        self.insert_impl(index, value, |_, _| true).map(
            |InsertError { value, error }| match error {
                InsertErrorKind::IndexOutOfBounds(_) => value,
                InsertErrorKind::ValueNotUnique => unsafe {
                    unreachable_dbg!("we know the added value was not tested for uniqueness")
                },
            },
        )
    }

    /// Pushes the (non-unique) `value` to the end of these [`EntryValues`].
    fn push(&mut self, value: V) {
        let none_ = self.push_impl(value, |_, _| true);
        debug_assert!(none_.is_none(), "`push()` should never fail");
    }

    fn len_nonzero(&self) -> NonZeroUsize {
        debug_assert!(self.len() > 0, "`EntryValues` should never be empty");
        unsafe { NonZeroUsize::new_unchecked(self.len()) }
    }

    /// `is_unique()` is passed (a slice of) existing values in the `entry` and the (reference to the) inserted `value`
    /// and returns `true` if the `value` is not contained in the entry (or is, but we allow duplicates) and must be inserted;
    /// or `false` otherwise, and the `value` must not be inserted.
    ///
    /// Done this way to avoid a `V: PartialEq` bound and delegate comparison to the caller instead.
    fn insert_impl<F>(&mut self, index: usize, value: V, is_unique: F) -> Option<InsertError<V>>
    where
        F: FnOnce(&[V], &V) -> bool,
    {
        match self {
            // A single value for this key - transform into `Multiple`.
            EntryValues::One(existing) => {
                if index > 1 {
                    return Some(InsertError::index_out_of_bounds(value, self.len_nonzero()));
                }

                if !is_unique(slice::from_ref(existing), &value) {
                    return Some(InsertError::value_non_unique(value));
                }

                // Replace the existing single value with an empty `Vec`, preallocated for 2 values,
                // and then insert the existing and new values to it.
                // This avoids removing/re-inserting the hashmap entry.
                // TODO: is there a way to do it without `unsafe`?
                let existing =
                    std::mem::replace(self, EntryValues::Multiple(Vec::with_capacity(2)));
                if let EntryValues::Multiple(values) = self {
                    debug_assert!(values.is_empty());
                    if let EntryValues::One(existing) = existing {
                        values.push(existing);
                        let none_ = vec_insert(values, index, value);
                        debug_assert!(
                            none_.is_none(),
                            "we know `index` is in bounds and `vec_insert()` should not fail"
                        );
                    } else {
                        unsafe { unreachable_dbg!("we know the existing value was `One`") };
                    }
                } else {
                    unsafe { unreachable_dbg!("we know the just inserted value was `Multiple`") };
                }

                None
            }
            // `Multiple` values for this key - just (try to) insert into the vec.
            EntryValues::Multiple(existing) => {
                let len = existing.len();
                debug_assert!(len >= 2);

                // Explicitly checking whether index is in bounds before testing for uniqueness.
                if index > len {
                    return Some(InsertError::index_out_of_bounds(value, self.len_nonzero()));
                }

                if is_unique(existing, &value) {
                    let none_ = vec_insert(existing, index, value);
                    debug_assert!(
                        none_.is_none(),
                        "we know `index` is in bounds and `vec_insert()` should not fail"
                    );
                    None
                } else {
                    Some(InsertError::value_non_unique(value))
                }
            }
        }
    }

    /// `is_unique()` is passed (a slice of) existing values in the `entry` and the (reference to the) inserted `value`
    /// and returns `true` if the `value` is not contained in the entry (or is, but we allow duplicates) and must be inserted;
    /// or `false` otherwise, and the `value` must not be inserted.
    ///
    /// Done this way to avoid a `V: PartialEq` bound and delegate comparison to the caller instead.
    ///
    /// Returns the `value` back if it was not inserted because `is_unique` returned `false`.
    fn push_impl<F>(&mut self, value: V, is_unique: F) -> Option<V>
    where
        F: FnOnce(&[V], &V) -> bool,
    {
        self.insert_impl(self.len(), value, is_unique)
            .map(|InsertError { value, error }| match error {
                InsertErrorKind::IndexOutOfBounds(_) => unsafe {
                    unreachable_dbg!(
                        "`push()` inserts at the end of the array and should never fail"
                    )
                },
                InsertErrorKind::ValueNotUnique => value,
            })
    }

    fn remove_impl(self, index: usize, swap: bool) -> Result<EntryValuesRemoveResult<V>, Self> {
        match self {
            EntryValues::One(value) => {
                if index == 0 {
                    Ok(EntryValuesRemoveResult::last(value))
                } else {
                    Err(Self::One(value))
                }
            }
            EntryValues::Multiple(mut values) => {
                debug_assert!(values.len() >= 2);
                match if swap {
                    vec_swap_remove(&mut values, index)
                } else {
                    vec_remove(&mut values, index)
                } {
                    Some(value) => Ok(EntryValuesRemoveResult::non_last(
                        value,
                        Self::multiple_to_one(values),
                    )),
                    None => Err(EntryValues::Multiple(values)),
                }
            }
        }
    }

    fn retain<F>(self, mut f: F) -> Option<Self>
    where
        F: FnMut(&mut V) -> bool,
    {
        match self {
            EntryValues::One(mut value) => {
                if f(&mut value) {
                    Some(EntryValues::One(value))
                } else {
                    None
                }
            }
            EntryValues::Multiple(mut values) => {
                values.retain_mut(f);
                if values.is_empty() {
                    None
                } else {
                    Some(Self::multiple_to_one(values))
                }
            }
        }
    }

    fn multiple_to_one(mut values: Vec<V>) -> Self {
        debug_assert!(!values.is_empty());
        if values.len() == 1 {
            Self::One(unsafe {
                values
                    .pop()
                    .unwrap_unchecked_dbg_msg("we know there's exactly one element in the `Vec`")
            })
        } else {
            Self::Multiple(values)
        }
    }
}

impl<V: PartialEq> EntryValues<V> {
    /// Tries to insert the (unique) `value` at `index` into these [`EntryValues`].
    /// Returns `None` on success, else returns the error.
    fn insert_unique(&mut self, index: usize, value: V) -> Option<InsertError<V>> {
        self.insert_impl(index, value, |values, value| !values.contains(value))
    }

    /// Pushes the (unique) `value` to the end of these [`EntryValues`].
    /// Returns `None` on success, else returns the `value` back if it is not unique.
    fn push_unique(&mut self, value: V) -> Option<V> {
        self.push_impl(value, |values, value| !values.contains(value))
    }
}

impl<T> Deref for EntryValues<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        match self {
            EntryValues::One(value) => slice::from_ref(value),
            EntryValues::Multiple(values) => values,
        }
    }
}

impl<T> DerefMut for EntryValues<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            EntryValues::One(value) => slice::from_mut(value),
            EntryValues::Multiple(values) => values,
        }
    }
}

/// A simple multimap wrapper around a [`HashMap`],
/// optimized for the usual case of having one value per key with no overhead
/// (i.e. a [`Vec`] is not allocated to hold a single value),
/// but which does support multiple (potentially non-unique) values by storing them in a [`Vec`] when necessary.
pub struct MultiMap<K, V, S = RandomState> {
    inner: HashMap<K, EntryValues<V>, S>,
    /// Used to keep track of all inserted values, not just keys via `inner.len()`.
    num_values: usize,
}

impl<K, V> MultiMap<K, V, RandomState> {
    /// See [`HashMap::new()`](HashMap::new).
    #[inline]
    #[must_use]
    pub fn new() -> MultiMap<K, V, RandomState> {
        Default::default()
    }

    /// See [`HashMap::with_capacity()`](HashMap::with_capacity).
    #[inline]
    #[must_use]
    pub fn with_capacity(capacity: usize) -> MultiMap<K, V, RandomState> {
        MultiMap::with_capacity_and_hasher(capacity, Default::default())
    }
}

impl<K, V, S> MultiMap<K, V, S> {
    /// See [`HashMap::with_hasher()`](HashMap::with_hasher).
    #[inline]
    #[must_use]
    pub fn with_hasher(hash_builder: S) -> MultiMap<K, V, S> {
        MultiMap {
            inner: HashMap::with_hasher(hash_builder),
            num_values: 0,
        }
    }

    /// See [`HashMap::with_capacity_and_hasher()`](HashMap::with_capacity_and_hasher).
    #[inline]
    #[must_use]
    pub fn with_capacity_and_hasher(capacity: usize, hasher: S) -> MultiMap<K, V, S> {
        MultiMap {
            inner: HashMap::with_capacity_and_hasher(capacity, hasher),
            num_values: 0,
        }
    }

    /// See [`HashMap::capacity()`](HashMap::capacity).
    #[inline]
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// See [`HashMap::keys()`](HashMap::keys).
    ///
    /// # Examples
    ///
    /// ```
    /// use minimultimap::MultiMap;
    ///
    /// let pairs = [(7, "foo"), (9, "bar"), (7, "baz")];
    ///
    /// let mmap = MultiMap::from(pairs);
    /// assert_eq!(mmap.len(), 2);
    /// assert_eq!(mmap.multi_len(), 3);
    ///
    /// let keys = mmap.keys();
    ///
    /// assert_eq!(keys.len(), 2);
    ///
    /// for k in keys {
    ///     assert!(*k == 7 || *k == 9);
    /// }
    /// ```
    pub fn keys(&self) -> Keys<'_, K, EntryValues<V>> {
        self.inner.keys()
    }

    /// See [`HashMap::into_keys()`](HashMap::into_keys).
    ///
    /// # Examples
    ///
    /// ```
    /// use minimultimap::MultiMap;
    ///
    /// let pairs = [(7, "foo"), (9, "bar"), (7, "baz")];
    ///
    /// let mmap = MultiMap::from(pairs);
    /// assert_eq!(mmap.len(), 2);
    /// assert_eq!(mmap.multi_len(), 3);
    ///
    /// let mut num_keys = 0;
    ///
    /// for k in mmap.into_keys() {
    ///     assert!(k == 7 || k == 9);
    ///     num_keys += 1;
    /// }
    ///
    /// assert_eq!(num_keys, 2);
    /// ```
    #[inline]
    pub fn into_keys(self) -> IntoKeys<K, EntryValues<V>> {
        self.inner.into_keys()
    }

    /// See [`HashMap::values()`](HashMap::values).
    ///
    /// # Examples
    ///
    /// ```
    /// use minimultimap::{ MultiMap, EntryValues };
    ///
    /// let pairs = [(7, "foo"), (9, "bar"), (7, "baz")];
    ///
    /// let mmap = MultiMap::from(pairs);
    /// assert_eq!(mmap.len(), 2);
    /// assert_eq!(mmap.multi_len(), 3);
    ///
    /// let values = mmap.values();
    ///
    /// assert_eq!(values.len(), 2);
    ///
    /// for v in values {
    ///     assert!(
    ///         *v == EntryValues::Multiple(vec!["foo", "baz"]) ||
    ///         *v == EntryValues::One("bar")
    ///     );
    /// }
    /// ```
    pub fn values(&self) -> Values<'_, K, EntryValues<V>> {
        self.inner.values()
    }

    /// See [`HashMap::values_mut()`](HashMap::values_mut).
    ///
    /// # Examples
    ///
    /// ```
    /// use minimultimap::{ MultiMap, EntryValues };
    ///
    /// let pairs = [(7, "foo"), (9, "bar"), (7, "baz")];
    ///
    /// let mut mmap = MultiMap::from(pairs);
    /// assert_eq!(mmap.len(), 2);
    /// assert_eq!(mmap.multi_len(), 3);
    ///
    /// let values = mmap.values_mut();
    ///
    /// assert_eq!(values.len(), 2);
    ///
    /// for v in values {
    ///     assert!(
    ///         *v == EntryValues::Multiple(vec!["foo", "baz"]) ||
    ///         *v == EntryValues::One("bar")
    ///     );
    ///
    ///     v[0] = "bob";
    /// }
    ///
    /// for v in mmap.values() {
    ///     assert!(
    ///         *v == EntryValues::Multiple(vec!["bob", "baz"]) ||
    ///         *v == EntryValues::One("bob")
    ///     );
    /// }
    /// ```
    pub fn values_mut(&mut self) -> ValuesMut<'_, K, EntryValues<V>> {
        self.inner.values_mut()
    }

    /// See [`HashMap::into_values()`](HashMap::into_values).
    ///
    /// # Examples
    ///
    /// ```
    /// use minimultimap::{ MultiMap, EntryValues };
    ///
    /// let pairs = [(7, "foo"), (9, "bar"), (7, "baz")];
    ///
    /// let mmap = MultiMap::from(pairs);
    /// assert_eq!(mmap.len(), 2);
    /// assert_eq!(mmap.multi_len(), 3);
    ///
    /// let mut num_values = 0;
    ///
    /// for v in mmap.into_values() {
    ///     assert!(
    ///         v == EntryValues::Multiple(vec!["foo", "baz"]) ||
    ///         v == EntryValues::One("bar")
    ///     );
    ///     num_values += 1;
    /// }
    ///
    /// assert_eq!(num_values, 2);
    /// ```
    #[inline]
    pub fn into_values(self) -> IntoValues<K, EntryValues<V>> {
        self.inner.into_values()
    }

    /// See [`HashMap::iter()`](HashMap::iter).
    ///
    /// # Examples
    ///
    /// ```
    /// use minimultimap::MultiMap;
    ///
    /// let pairs = [(7, "foo"), (9, "bar"), (7, "baz")];
    ///
    /// let mut mmap = MultiMap::from(pairs);
    /// assert_eq!(mmap.len(), 2);
    /// assert_eq!(mmap.multi_len(), 3);
    ///
    /// let iter = mmap.iter();
    /// assert_eq!(iter.len(), 2);
    ///
    /// for (k, vs) in iter {
    ///     for v in vs.iter() {
    ///         assert!(pairs.contains(&(*k, *v)));
    ///     }
    /// }
    /// ```
    pub fn iter(&self) -> Iter<'_, K, EntryValues<V>> {
        self.inner.iter()
    }

    /// See [`HashMap::iter_mut()`](HashMap::iter_mut).
    ///
    /// # Examples
    ///
    /// ```
    /// use minimultimap::{ MultiMap, EntryValues };
    ///
    /// let pairs = [(7, "foo".to_string()), (9, "bar".to_string()), (7, "baz".to_string())];
    ///
    /// let mut mmap = MultiMap::from(pairs.clone());
    /// assert_eq!(mmap.len(), 2);
    /// assert_eq!(mmap.multi_len(), 3);
    ///
    /// let iter = mmap.iter_mut();
    /// assert_eq!(iter.len(), 2);
    ///
    /// for (k, vs) in iter {
    ///     for v in vs.iter_mut() {
    ///         assert!(pairs.contains(&(*k, v.clone())));
    ///         v.push_str(&format!("_{}", *k));
    ///     }
    /// }
    ///
    /// for v in mmap.values() {
    ///     assert!(
    ///         *v == EntryValues::Multiple(vec!["foo_7".to_string(), "baz_7".to_string()]) ||
    ///         *v == EntryValues::One("bar_9".to_string())
    ///     );
    /// }
    /// ```
    pub fn iter_mut(&mut self) -> IterMut<'_, K, EntryValues<V>> {
        self.inner.iter_mut()
    }

    /// Returns an iterator over (references to) all key-value tuples in the [`MultiMap`], in unspecified order.
    ///
    /// Similar to [`iter()`](Self::iter), but the same key may be returned multiple times with single values which
    /// might or might not be unique, depending on how they were inserted into the [`MultiMap`].
    ///
    /// # Examples
    ///
    /// ```
    /// use minimultimap::MultiMap;
    ///
    /// let pairs = [(7, "foo"), (9, "bar"), (7, "baz")];
    ///
    /// let mut mmap = MultiMap::from(pairs);
    /// assert_eq!(mmap.len(), 2);
    /// assert_eq!(mmap.multi_len(), 3);
    ///
    /// let iter = mmap.multi_iter();
    /// assert_eq!(iter.len(), 3);
    ///
    /// for (k, v) in iter {
    ///     assert!(pairs.contains(&(*k, *v)));
    /// }
    /// ```
    pub fn multi_iter(&self) -> MultiIter<'_, K, V> {
        MultiIter::new(self.iter(), self.num_values)
    }

    /// Returns an iterator over (mutable references to) all key-value tuples in the [`MultiMap`], in unspecified order.
    ///
    /// Similar to [`iter_mut()`](Self::iter_mut), but the same key may be returned multiple times with single values which
    /// might or might not be unique, depending on how they were inserted into the [`MultiMap`].
    ///
    /// # Examples
    ///
    /// ```
    /// use minimultimap::{ MultiMap, EntryValues };
    ///
    /// let pairs = [(7, "foo".to_string()), (9, "bar".to_string()), (7, "baz".to_string())];
    ///
    /// let mut mmap = MultiMap::from(pairs.clone());
    /// assert_eq!(mmap.len(), 2);
    /// assert_eq!(mmap.multi_len(), 3);
    ///
    /// let iter = mmap.multi_iter_mut();
    /// assert_eq!(iter.len(), 3);
    ///
    /// for (k, v) in iter {
    ///     assert!(pairs.contains(&(*k, v.clone())));
    ///     v.push_str(&format!("_{}", *k));
    /// }
    ///
    /// for v in mmap.values() {
    ///     assert!(
    ///         *v == EntryValues::Multiple(vec!["foo_7".to_string(), "baz_7".to_string()]) ||
    ///         *v == EntryValues::One("bar_9".to_string())
    ///     );
    /// }
    /// ```
    pub fn multi_iter_mut(&mut self) -> MultiIterMut<'_, K, V> {
        let num_values = self.num_values;
        MultiIterMut::new(self.iter_mut(), num_values)
    }

    /// See [`HashMap::len()`](HashMap::len), but replace "elements" with "keys".
    ///
    /// # Examples
    ///
    /// ```
    /// use minimultimap::MultiMap;
    ///
    /// let pairs = [(7, "foo".to_string()), (9, "bar".to_string()), (7, "baz".to_string())];
    ///
    /// let mut mmap = MultiMap::from(pairs.clone());
    /// assert_eq!(mmap.len(), 2);
    /// assert_eq!(mmap.multi_len(), 3);
    /// ```
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Unlike [`len()`](Self::len), returns the total number of values on the [`MultiMap`].
    ///
    /// # Examples
    ///
    /// ```
    /// use minimultimap::MultiMap;
    ///
    /// let pairs = [(7, "foo".to_string()), (9, "bar".to_string()), (7, "baz".to_string())];
    ///
    /// let mut mmap = MultiMap::from(pairs.clone());
    /// assert_eq!(mmap.len(), 2);
    /// assert_eq!(mmap.multi_len(), 3);
    /// ```
    #[inline]
    pub fn multi_len(&self) -> usize {
        self.num_values
    }

    /// See [`HashMap::is_empty()`](HashMap::is_empty), but replace "elements" with "keys".
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// See [`HashMap::drain()`](HashMap::drain).
    ///
    /// # Examples
    ///
    /// ```
    /// use minimultimap::MultiMap;
    ///
    /// let mut mmap = MultiMap::new();
    /// mmap.add(1, "a");
    /// mmap.add(2, "b");
    /// mmap.add(1, "c");
    ///
    /// assert_eq!(mmap.len(), 2);
    /// assert_eq!(mmap.multi_len(), 3);
    ///
    /// let capacity = mmap.capacity();
    ///
    /// for (k, vs) in mmap.drain().take(2) {
    ///     assert!(
    ///         ((k == 1) && (*vs == ["a", "c"]))
    ///         || ((k == 2) && (*vs == ["b"]))
    ///     );
    /// }
    ///
    /// assert!(mmap.is_empty());
    /// assert_eq!(mmap.multi_len(), 0);
    /// assert_eq!(mmap.capacity(), capacity);
    /// ```
    #[inline]
    pub fn drain(&mut self) -> Drain<'_, K, EntryValues<V>> {
        let drain = self.inner.drain();
        self.num_values = 0;
        drain
    }

    /// See [`HashMap::retain()`](HashMap::retain), but replace "elements" with "keys".
    ///
    /// # Examples
    ///
    /// ```
    /// use minimultimap::MultiMap;
    ///
    /// let mut mmap = MultiMap::new();
    /// mmap.add(1, "a");
    /// mmap.add(2, "b");
    /// mmap.add(1, "c");
    ///
    /// assert_eq!(mmap.len(), 2);
    /// assert_eq!(mmap.multi_len(), 3);
    ///
    /// mmap.retain(|k, vs| *k != 1);
    ///
    /// assert_eq!(mmap.len(), 1);
    /// assert_eq!(mmap.multi_len(), 1);
    ///
    /// assert!(!mmap.contains_key(&1));
    /// assert_eq!(mmap[&2], ["b"]);
    /// ```
    #[inline]
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&K, &mut [V]) -> bool,
    {
        self.inner.retain(|k, vs| {
            let retain = f(k, vs.deref_mut());
            if !retain {
                let len = vs.len();
                debug_assert!(len > 0);
                debug_assert!(self.num_values >= len);
                self.num_values -= len;
            }
            retain
        })
    }

    /// Similar to [`retain()`](Self::retain), but the same key may be passed to the predicate multiple times with single values which
    /// might or might not be unique, depending on how they were inserted into the [`MultiMap`].
    ///
    /// # Examples
    ///
    /// ```
    /// use minimultimap::MultiMap;
    ///
    /// let mut mmap = MultiMap::new();
    /// mmap.add(1, "a");
    /// mmap.add(2, "b");
    /// mmap.add(1, "c");
    /// mmap.add(3, "d");
    /// mmap.add(3, "e");
    ///
    /// assert_eq!(mmap.len(), 3);
    /// assert_eq!(mmap.multi_len(), 5);
    ///
    /// mmap.multi_retain(|k, v| *k == 3 || *v == "a");
    ///
    /// assert_eq!(mmap.len(), 2);
    /// assert_eq!(mmap.multi_len(), 3);
    ///
    /// assert_eq!(mmap[&1], ["a"]);
    /// assert!(!mmap.contains_key(&2));
    /// assert_eq!(mmap[&3], ["d", "e"]);
    /// ```
    #[inline]
    pub fn multi_retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&K, &mut V) -> bool,
    {
        self.inner.retain(|k, vs| {
            let len = vs.len();
            debug_assert!(len > 0);

            // Temporarily swap the existing value(s) with a dummy.
            let empty = EntryValues::Multiple(vec![]);
            let previous = std::mem::replace(vs, empty);

            match previous.retain(|v| f(k, v)) {
                // Some value(s) retained. Put them back, retain the entry values.
                Some(remaining) => {
                    let remaining_len = remaining.len();
                    debug_assert!(remaining_len > 0);
                    debug_assert!(remaining_len <= len);

                    let removed_len = len - remaining_len;
                    debug_assert!(self.num_values >= removed_len);
                    self.num_values -= removed_len;

                    let empty_ = std::mem::replace(vs, remaining);
                    debug_assert!(matches!(empty_, EntryValues::Multiple(..)));
                    debug_assert!(empty_.is_empty());

                    true
                }
                // No values retained. Do not retain the entry values.
                None => {
                    debug_assert!(self.num_values >= len);
                    self.num_values -= len;

                    false
                }
            }
        })
    }

    /// See [`HashMap::clear()`](HashMap::clear).
    ///
    /// # Examples
    ///
    /// ```
    /// use minimultimap::MultiMap;
    ///
    /// let pairs = [(7, "foo".to_string()), (9, "bar".to_string()), (7, "baz".to_string())];
    ///
    /// let mut mmap = MultiMap::from(pairs.clone());
    /// assert!(!mmap.is_empty());
    /// assert_eq!(mmap.len(), 2);
    /// assert_eq!(mmap.multi_len(), 3);
    ///
    /// mmap.clear();
    ///
    /// assert!(mmap.is_empty());
    /// assert_eq!(mmap.len(), 0);
    /// assert_eq!(mmap.multi_len(), 0);
    /// ```
    #[inline]
    pub fn clear(&mut self) {
        self.inner.clear();
        self.num_values = 0;
    }

    /// See [`HashMap::hasher()`](HashMap::hasher).
    #[inline]
    pub fn hasher(&self) -> &S {
        self.inner.hasher()
    }

    fn dec_num_values(&mut self, num: usize) {
        debug_assert!(self.num_values >= num);
        self.num_values -= num;
    }
}

impl<K, V, S> MultiMap<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    /// See [`HashMap::reserve()`](HashMap::reserve), but replace "elements" with "keys".
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.inner.reserve(additional)
    }

    /// See [`HashMap::try_reserve()`](HashMap::try_reserve), but replace "elements" with "keys".
    #[inline]
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.inner.try_reserve(additional)
    }

    /// See [`HashMap::shrink_to_fit()`](HashMap::shrink_to_fit).
    #[inline]
    pub fn shrink_to_fit(&mut self) {
        self.inner.shrink_to_fit();
    }

    /// See [`HashMap::shrink_to()`](HashMap::shrink_to).
    #[inline]
    pub fn shrink_to(&mut self, min_capacity: usize) {
        self.inner.shrink_to(min_capacity);
    }

    /// See [`HashMap::entry()`](HashMap::entry).
    ///
    /// # Examples
    ///
    /// ```
    /// use minimultimap::{ MultiMap, Entry };
    ///
    /// let mut mmap = MultiMap::<i32, &str>::from_iter([(7, "foo"), (7, "bar"), (9, "baz")].into_iter());
    /// assert_eq!(mmap.len(), 2);
    /// assert_eq!(mmap.multi_len(), 3);
    ///
    /// assert!(matches!(mmap.entry(11), Entry::Vacant(..)));
    ///
    /// match mmap.entry(7) {
    ///     Entry::Occupied(entry) => {
    ///         assert_eq!(entry.get(), &["foo", "bar"]);
    ///     },
    ///     Entry::Vacant(_) => unreachable!(),
    /// }
    ///
    /// match mmap.entry(9) {
    ///     Entry::Occupied(entry) => {
    ///         assert_eq!(entry.get(), &["baz"]);
    ///     },
    ///     Entry::Vacant(_) => unreachable!(),
    /// }
    /// ```
    #[inline]
    pub fn entry(&mut self, key: K) -> Entry<'_, K, V> {
        let num_values = unsafe { NonNull::new_unchecked(&mut self.num_values as *mut _) };
        match self.inner.entry(key) {
            HashMapEntry::Occupied(inner) => Entry::Occupied(OccupiedEntry { inner, num_values }),
            HashMapEntry::Vacant(inner) => Entry::Vacant(VacantEntry { inner, num_values }),
        }
    }

    /// See [`HashMap::get()`](HashMap::get).
    #[inline]
    pub fn get<Q: ?Sized>(&self, k: &Q) -> Option<&[V]>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.inner.get(k).map(EntryValues::deref)
    }

    /// Like [`get()`](Self::get), but returns an iterator over references to all values asociated with the key `k`,
    /// empty if the key is not contained in the map.
    ///
    /// # Examples
    ///
    /// ```
    /// use minimultimap::MultiMap;
    ///
    /// let mmap = MultiMap::<i32, &'static str>::from_iter([(7, "foo"), (7, "bar"), (9, "baz")].into_iter());
    /// assert_eq!(mmap.len(), 2);
    /// assert_eq!(mmap.multi_len(), 3);
    ///
    /// assert!(mmap.get_iter(&7).eq(["foo", "bar"].iter()));
    /// assert_eq!(mmap.get_iter(&8).len(), 0);
    /// assert!(mmap.get_iter(&9).eq(["baz"].iter()));
    /// ```
    pub fn get_iter<Q: ?Sized>(&self, k: &Q) -> SliceIter<'_, V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.get(k).map(<[V]>::iter).unwrap_or([].iter())
    }

    /// See [`HashMap::get_key_value()`](HashMap::get_key_value).
    #[inline]
    pub fn get_key_value<Q: ?Sized>(&self, k: &Q) -> Option<(&K, &[V])>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.inner.get_key_value(k).map(|(k, v)| (k, v.deref()))
    }

    /// See [`HashMap::contains_key()`](HashMap::contains_key).
    #[inline]
    pub fn contains_key<Q: ?Sized>(&self, k: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.inner.contains_key(k)
    }

    /// See [`HashMap::get_mut()`](HashMap::get_mut).
    #[inline]
    pub fn get_mut<Q: ?Sized>(&mut self, k: &Q) -> Option<&mut [V]>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.inner.get_mut(k).map(EntryValues::deref_mut)
    }

    /// Like [`get_mut()`](Self::get_mut), but returns an iterator over mutable references to all values asociated with the key `k`,
    /// empty if the key is not contained in the map.
    ///
    /// # Examples
    ///
    /// ```
    /// use minimultimap::MultiMap;
    ///
    /// let mut mmap = MultiMap::<u8, String>::from_iter([(7, "hello".to_string()), (7, "goodbye".to_string()), (9, "foo".to_string())].into_iter());
    /// assert_eq!(mmap.len(), 2);
    /// assert_eq!(mmap.multi_len(), 3);
    ///
    /// assert!(mmap.get_iter_mut(&7).eq(["hello", "goodbye"].iter_mut()));
    /// assert_eq!(mmap.get_iter_mut(&8).len(), 0);
    /// assert!(mmap.get_iter_mut(&9).eq(["foo"].iter_mut()));
    ///
    /// for val in mmap.get_iter_mut(&7) {
    ///     val.push_str(" world");
    /// }
    ///
    /// assert!(mmap.get_iter_mut(&7).eq(["hello world", "goodbye world"].iter_mut()));
    /// ```
    pub fn get_iter_mut<Q: ?Sized>(&mut self, k: &Q) -> SliceIterMut<'_, V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.get_mut(k)
            .map(<[V]>::iter_mut)
            .unwrap_or([].iter_mut())
    }

    /// See [`HashMap::insert()`](HashMap::insert).
    ///
    /// Keeps the same semantics as [`HashMap::insert()`](HashMap::insert) for drop-in compatibility (i.e. replaces the value(s) at key `k` with a new value `v`),
    /// but would probably be better named `replace()`.
    ///
    /// Use [`add()`](Self::add) / [`add_unique()`](Self::add_unique) to actually add a new (non-unique / unique) value `v` for the key `k`.
    #[inline]
    pub fn insert(&mut self, k: K, v: V) -> Option<EntryValues<V>> {
        let exisiting = self.inner.insert(k, EntryValues::One(v));
        self.num_values += 1;
        exisiting
    }

    /// Adds the (non-unique) value `v` to the list of values for the key `k`.
    ///
    /// If the key `k` already existed in the [`MultiMap`],
    /// returns the (non-zero) number of values associated with this key before this call.
    ///
    /// Otherwise inserts the key-value pair and returns `None`.
    ///
    /// NOTE: use [`add_unique()`](Self::add_unique) to only add unique values for the key `k`.
    #[inline]
    pub fn add(&mut self, k: K, v: V) -> Option<NonZeroUsize> {
        match self.entry(k) {
            Entry::Occupied(mut e) => {
                let len = e.get_entry_values().len_nonzero();
                e.push(v);
                Some(len)
            }
            Entry::Vacant(e) => {
                e.insert(v);
                None
            }
        }
    }

    /// See [`HashMap::remove()`](HashMap::remove).
    ///
    /// # Examples
    ///
    /// ```
    /// use minimultimap::{ MultiMap, EntryValues };
    ///
    /// let mut mmap = MultiMap::<i32, String>::from_iter([(7, "foo".to_string()), (7, "bar".to_string()), (9, "baz".to_string())].into_iter());
    /// assert_eq!(mmap.len(), 2);
    /// assert_eq!(mmap.multi_len(), 3);
    ///
    /// let foobar = mmap.remove(&7).unwrap();
    /// assert_eq!(foobar, EntryValues::Multiple(vec!["foo".to_string(), "bar".to_string()]));
    /// assert_eq!(mmap.len(), 1);
    /// assert_eq!(mmap.multi_len(), 1);
    /// ```
    #[inline]
    pub fn remove<Q: ?Sized>(&mut self, k: &Q) -> Option<EntryValues<V>>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let removed = self.inner.remove(k);
        if let Some(removed) = removed.as_ref() {
            self.dec_num_values(removed.len());
        }
        removed
    }

    /// See [`HashMap::remove_entry()`](HashMap::remove_entry).
    ///
    /// # Examples
    ///
    /// ```
    /// use minimultimap::{ MultiMap, EntryValues };
    ///
    /// let mut mmap = MultiMap::<u32, String>::from_iter([(7, "foo".to_string()), (7, "bar".to_string()), (9, "baz".to_string())].into_iter());
    /// assert_eq!(mmap.len(), 2);
    /// assert_eq!(mmap.multi_len(), 3);
    ///
    /// let foobar = mmap.remove_entry(&7).unwrap();
    /// assert_eq!(foobar.0, 7);
    /// assert_eq!(foobar.1, EntryValues::Multiple(vec!["foo".to_string(), "bar".to_string()]));
    /// assert_eq!(mmap.len(), 1);
    /// assert_eq!(mmap.multi_len(), 1);
    /// ```
    #[inline]
    pub fn remove_entry<Q: ?Sized>(&mut self, k: &Q) -> Option<(K, EntryValues<V>)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let removed = self.inner.remove_entry(k);
        if let Some((_, removed)) = removed.as_ref() {
            self.dec_num_values(removed.len());
        }
        removed
    }
}

/// Result of [`MultiMap::add_unique()`](MultiMap::add_unique).
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum AddUniqueResult<V> {
    /// Key did not exist in the map - key-value pair was added.
    KeyDidNotExist,
    /// Key existed in the map, but the passed value was not unique - the value is returned back,
    /// as well as the (non-zero) number of values associated with this key before the call.
    ValueNotUnique((V, NonZeroUsize)),
    /// Key existed in the map, and the passed value was unique - value was added,
    /// and the (non-zero) number of values associated with this key before the call is returned back.
    ValueUnique(NonZeroUsize),
}

impl<K, V, S> MultiMap<K, V, S>
where
    K: Eq + Hash,
    V: PartialEq,
    S: BuildHasher,
{
    /// Tries to add a (unique) value `v` to the list of values for the key `k`.
    ///
    /// NOTE: value uniqueness is checked by linearly scanning all values associated with the key, which might be suboptimal.
    ///
    /// NOTE: use [`add()`](Self::add) to add non-unique values for the key `k`.
    ///
    /// # Examples
    ///
    /// ```
    /// use { std::num::NonZeroUsize, minimultimap::{ MultiMap, AddUniqueResult } };
    ///
    /// let mut mmap = MultiMap::<i32, String>::default();
    ///
    /// assert_eq!(mmap.add_unique(7, "foo".to_string()), AddUniqueResult::KeyDidNotExist);
    /// assert_eq!(mmap[&7], ["foo".to_string()]);
    /// assert_eq!(mmap.len(), 1);
    /// assert_eq!(mmap.multi_len(), 1);
    ///
    /// assert_eq!(mmap.add_unique(7, "foo".to_string()), AddUniqueResult::ValueNotUnique(("foo".to_string(), NonZeroUsize::new(1).unwrap())));
    /// assert_eq!(mmap[&7], ["foo".to_string()]);
    /// assert_eq!(mmap.len(), 1);
    /// assert_eq!(mmap.multi_len(), 1);
    ///
    /// assert_eq!(mmap.add_unique(7, "bar".to_string()), AddUniqueResult::ValueUnique(NonZeroUsize::new(1).unwrap()));
    /// assert_eq!(mmap[&7], ["foo".to_string(), "bar".to_string()]);
    /// assert_eq!(mmap.len(), 1);
    /// assert_eq!(mmap.multi_len(), 2);
    /// ```
    #[inline]
    pub fn add_unique(&mut self, k: K, v: V) -> AddUniqueResult<V> {
        match self.entry(k) {
            Entry::Occupied(mut e) => {
                let len = e.get_entry_values().len_nonzero();
                if let Some(non_unique) = e.push_unique(v) {
                    AddUniqueResult::ValueNotUnique((non_unique, len))
                } else {
                    AddUniqueResult::ValueUnique(len)
                }
            }
            Entry::Vacant(e) => {
                e.insert(v);
                AddUniqueResult::KeyDidNotExist
            }
        }
    }
}

impl<K, V, S> Clone for MultiMap<K, V, S>
where
    K: Clone,
    V: Clone,
    S: Clone,
{
    #[inline]
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            num_values: self.num_values,
        }
    }

    #[inline]
    fn clone_from(&mut self, other: &Self) {
        self.inner.clone_from(&other.inner);
        self.num_values = other.num_values;
    }
}

impl<K, V, S> PartialEq for MultiMap<K, V, S>
where
    K: Eq + Hash,
    V: PartialEq,
    S: BuildHasher,
{
    fn eq(&self, other: &MultiMap<K, V, S>) -> bool {
        if self.num_values != other.num_values {
            return false;
        }

        self.inner.eq(&other.inner)
    }
}

impl<K, V, S> Eq for MultiMap<K, V, S>
where
    K: Eq + Hash,
    V: Eq,
    S: BuildHasher,
{
}

impl<K, V, S> Debug for MultiMap<K, V, S>
where
    K: Debug,
    V: Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl<K, V, S> Default for MultiMap<K, V, S>
where
    S: Default,
{
    /// See [`HashMap::default()`](HashMap::default).
    #[inline]
    fn default() -> MultiMap<K, V, S> {
        MultiMap::with_hasher(Default::default())
    }
}

impl<K, Q: ?Sized, V, S> Index<&Q> for MultiMap<K, V, S>
where
    K: Eq + Hash + Borrow<Q>,
    Q: Eq + Hash,
    S: BuildHasher,
{
    type Output = [V];

    /// See [`HashMap::index()`](HashMap::index).
    #[inline]
    fn index(&self, key: &Q) -> &[V] {
        self.get(key).expect("no entry found for key")
    }
}

impl<K, V, const N: usize> From<[(K, V); N]> for MultiMap<K, V, RandomState>
where
    K: Eq + Hash,
{
    fn from(arr: [(K, V); N]) -> Self {
        Self::from_iter(arr)
    }
}

/// See [`std::collections::hash_map::Entry`].
///
/// This type is needed (instead of just using the [`std::collections::hash_map::Entry`]) to correctly handle the number of values in the [`MultiMap`]
/// when inserting/removing through the entry API.
pub enum Entry<'a, K: 'a, V: 'a> {
    /// An occupied entry.
    Occupied(OccupiedEntry<'a, K, V>),
    /// A vacant entry.
    Vacant(VacantEntry<'a, K, V>),
}

impl<K: Debug, V: Debug> Debug for Entry<'_, K, V> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Entry::Vacant(v) => f.debug_tuple("Entry").field(v).finish(),
            Entry::Occupied(o) => f.debug_tuple("Entry").field(o).finish(),
        }
    }
}

/// See [`std::collections::hash_map::OccupiedEntry`].
///
/// This type is needed (instead of just using the [`std::collections::hash_map::OccupiedEntry`]) to correctly handle the number of values in the [`MultiMap`]
/// when inserting/removing through the entry API.
pub struct OccupiedEntry<'a, K: 'a, V: 'a> {
    inner: HashMapOccupiedEntry<'a, K, EntryValues<V>>,
    /// Pointer to the number of values in the map, needed to update it when adding / removing values.
    num_values: NonNull<usize>,
}

struct ValuesFormatter<'a, V>(&'a [V]);

impl<'a, V: Debug> Debug for ValuesFormatter<'a, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.len() > 1 {
            f.debug_list().entries(self.0).finish()
        } else {
            self.0[0].fmt(f)
        }
    }
}

impl<K: Debug, V: Debug> Debug for OccupiedEntry<'_, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OccupiedEntry")
            .field("key", self.key())
            .field(
                if self.get().len() > 1 {
                    "values"
                } else {
                    "value"
                },
                &ValuesFormatter(self.get()),
            )
            .finish_non_exhaustive()
    }
}

/// See [`std::collections::hash_map::VacantEntry`].
///
/// This type is needed (instead of just using the [`std::collections::hash_map::VacantEntry`]) to correctly handle the number of values in the [`MultiMap`]
/// when inserting through the entry API.
pub struct VacantEntry<'a, K: 'a, V: 'a> {
    inner: HashMapVacantEntry<'a, K, EntryValues<V>>,
    /// Pointer to the number of values in the map, needed to update it when inserting a value.
    num_values: NonNull<usize>,
}

impl<K: Debug, V> Debug for VacantEntry<'_, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("VacantEntry").field(self.key()).finish()
    }
}

impl<'a, K, V, S> IntoIterator for &'a MultiMap<K, V, S> {
    type Item = (&'a K, &'a EntryValues<V>);
    type IntoIter = Iter<'a, K, EntryValues<V>>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, K, V, S> IntoIterator for &'a mut MultiMap<K, V, S> {
    type Item = (&'a K, &'a mut EntryValues<V>);
    type IntoIter = IterMut<'a, K, EntryValues<V>>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<K, V, S> IntoIterator for MultiMap<K, V, S> {
    type Item = (K, EntryValues<V>);
    type IntoIter = IntoIter<K, EntryValues<V>>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.inner.into_iter()
    }
}

impl<'a, K, V> Entry<'a, K, V> {
    /// See [`std::collections::hash_map::Entry::or_insert`].
    #[inline]
    pub fn or_insert(self, default: V) -> &'a mut [V] {
        match self {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => slice::from_mut(entry.insert(default)),
        }
    }

    /// See [`std::collections::hash_map::Entry::or_insert_with`].
    #[inline]
    pub fn or_insert_with<F: FnOnce() -> V>(self, default: F) -> &'a mut [V] {
        match self {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => slice::from_mut(entry.insert(default())),
        }
    }

    /// See [`std::collections::hash_map::Entry::or_insert_with_key`].
    #[inline]
    pub fn or_insert_with_key<F: FnOnce(&K) -> V>(self, default: F) -> &'a mut [V] {
        match self {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                let value = default(entry.key());
                slice::from_mut(entry.insert(value))
            }
        }
    }

    /// See [`std::collections::hash_map::Entry::key`].
    #[inline]
    pub fn key(&self) -> &K {
        match self {
            Entry::Occupied(entry) => entry.key(),
            Entry::Vacant(entry) => entry.key(),
        }
    }

    /// See [`std::collections::hash_map::Entry::and_modify`].
    #[inline]
    pub fn and_modify<F>(self, f: F) -> Self
    where
        F: FnOnce(&mut [V]),
    {
        match self {
            Entry::Occupied(mut entry) => {
                f(entry.get_mut());
                Entry::Occupied(entry)
            }
            Entry::Vacant(entry) => Entry::Vacant(entry),
        }
    }
}

impl<'a, K, V: Default> Entry<'a, K, V> {
    /// See [`std::collections::hash_map::Entry::or_default`].
    #[inline]
    pub fn or_default(self) -> &'a mut [V] {
        match self {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => slice::from_mut(entry.insert(Default::default())),
        }
    }
}

/// Result returned when removing a value from an [`OccupiedEntry`].
pub struct RemoveResult<'a, K, V> {
    /// Removed value.
    pub removed: V,
    /// If the removed `value` was not last, the [`OccupiedEntry`]` with the remaining values.
    pub remaining: Option<OccupiedEntry<'a, K, V>>,
}

impl<'a, K, V> RemoveResult<'a, K, V> {
    fn last(removed: V) -> Self {
        Self {
            removed,
            remaining: None,
        }
    }

    fn not_last(removed: V, remaining: OccupiedEntry<'a, K, V>) -> Self {
        Self {
            removed,
            remaining: Some(remaining),
        }
    }
}

impl<'a, K, V> OccupiedEntry<'a, K, V> {
    /// See [`std::collections::hash_map::OccupiedEntry::key`].
    #[inline]
    pub fn key(&self) -> &K {
        self.inner.key()
    }

    /// See [`std::collections::hash_map::OccupiedEntry::remove_entry`].
    ///
    /// # Examples
    ///
    /// ```
    /// use minimultimap::{ MultiMap, Entry, EntryValues };
    ///
    /// let mut mmap = MultiMap::from([(7, "foo"), (9, "bar"), (7, "baz")]);
    /// assert_eq!(mmap.len(), 2);
    /// assert_eq!(mmap.multi_len(), 3);
    ///
    /// match mmap.entry(7) {
    ///     Entry::Occupied(entry) => {
    ///         assert_eq!(entry.remove_entry(), (7, EntryValues::Multiple(vec!["foo", "baz"])));
    ///     },
    ///     Entry::Vacant(_) => {},
    /// }
    ///
    /// assert!(!mmap.contains_key(&7));
    /// assert_eq!(mmap.len(), 1);
    /// assert_eq!(mmap.multi_len(), 1);
    /// ```
    #[inline]
    pub fn remove_entry(mut self) -> (K, EntryValues<V>) {
        let removed = self.inner.remove_entry();
        Self::dec_num_values(&mut self.num_values, removed.1.len());
        removed
    }

    /// See [`std::collections::hash_map::OccupiedEntry::get`].
    #[inline]
    pub fn get(&self) -> &[V] {
        self.inner.get()
    }

    /// See [`std::collections::hash_map::OccupiedEntry::get_mut`].
    #[inline]
    pub fn get_mut(&mut self) -> &mut [V] {
        self.inner.get_mut()
    }

    /// See [`std::collections::hash_map::OccupiedEntry::into_mut`].
    #[inline]
    pub fn into_mut(self) -> &'a mut [V] {
        self.inner.into_mut()
    }

    /// See [`std::collections::hash_map::OccupiedEntry::insert`].
    ///
    /// # Examples
    ///
    /// ```
    /// use minimultimap::{ MultiMap, Entry, EntryValues };
    ///
    /// let mut mmap = MultiMap::from([(7, "foo".to_string()), (9, "bar".to_string()), (7, "baz".to_string())]);
    /// assert_eq!(mmap.len(), 2);
    /// assert_eq!(mmap.multi_len(), 3);
    ///
    /// match mmap.entry(7) {
    ///     Entry::Occupied(mut entry) => {
    ///         assert_eq!(entry.insert("bob".to_string()), EntryValues::Multiple(vec!["foo".to_string(), "baz".to_string()]));
    ///     },
    ///     Entry::Vacant(_) => unreachable!(),
    /// }
    ///
    /// assert_eq!(mmap[&7], ["bob".to_string()]);
    /// assert_eq!(mmap.len(), 2);
    /// assert_eq!(mmap.multi_len(), 2);
    /// ```
    #[inline]
    pub fn insert(&mut self, value: V) -> EntryValues<V> {
        let previous = self.inner.insert(EntryValues::One(value));
        let num_previous = previous.len();
        debug_assert!(num_previous > 0);
        Self::dec_num_values(&mut self.num_values, num_previous - 1);
        previous
    }

    /// See [`std::collections::hash_map::OccupiedEntry::remove`].
    ///
    /// # Examples
    ///
    /// ```
    /// use minimultimap::{ MultiMap, Entry, EntryValues };
    ///
    /// let mut mmap = MultiMap::from([(7, "foo".to_string()), (9, "bar".to_string()), (7, "baz".to_string())]);
    /// assert_eq!(mmap.len(), 2);
    /// assert_eq!(mmap.multi_len(), 3);
    ///
    /// match mmap.entry(7) {
    ///     Entry::Occupied(entry) => {
    ///         assert_eq!(entry.remove(), EntryValues::Multiple(vec!["foo".to_string(), "baz".to_string()]));
    ///     },
    ///     Entry::Vacant(_) => unreachable!(),
    /// }
    ///
    /// assert!(!mmap.contains_key(&7));
    /// assert_eq!(mmap.len(), 1);
    /// assert_eq!(mmap.multi_len(), 1);
    /// ```
    #[inline]
    pub fn remove(mut self) -> EntryValues<V> {
        let removed = self.inner.remove();
        Self::dec_num_values(&mut self.num_values, removed.len());
        removed
    }

    /// Tries to insert the (non-unique) `value` at `index` into this entry.
    /// Returns `None` on success, else returns the `value` back if `index` is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use minimultimap::{ MultiMap, Entry };
    ///
    /// let mut mmap = MultiMap::from([(7, "foo".to_string()), (9, "bar".to_string())]);
    /// assert_eq!(mmap.len(), 2);
    /// assert_eq!(mmap.multi_len(), 2);
    ///
    /// match mmap.entry(7) {
    ///     Entry::Occupied(mut entry) => {
    ///         assert!(entry.insert_at(2, "baz".to_string()).is_some());
    ///         assert!(entry.insert_at(0, "baz".to_string()).is_none());
    ///     },
    ///     Entry::Vacant(_) => unreachable!(),
    /// }
    ///
    /// assert_eq!(mmap[&7], ["baz".to_string(), "foo".to_string()]);
    /// assert_eq!(mmap.len(), 2);
    /// assert_eq!(mmap.multi_len(), 3);
    /// ```
    #[inline]
    pub fn insert_at(&mut self, index: usize, value: V) -> Option<V> {
        self.inner.get_mut().insert(index, value).or_else(|| {
            self.inc_num_values(1);
            None
        })
    }

    /// Pushes the (non-unique) `value` to the end of this entry.
    ///
    /// # Examples
    ///
    /// ```
    /// use minimultimap::{ MultiMap, Entry };
    ///
    /// let mut mmap = MultiMap::from([(7, "foo".to_string()), (9, "bar".to_string())]);
    /// assert_eq!(mmap.len(), 2);
    /// assert_eq!(mmap.multi_len(), 2);
    ///
    /// match mmap.entry(7) {
    ///     Entry::Occupied(mut entry) => entry.push("baz".to_string()),
    ///     Entry::Vacant(_) => unreachable!(),
    /// }
    ///
    /// assert_eq!(mmap[&7], ["foo".to_string(), "baz".to_string()]);
    /// assert_eq!(mmap.len(), 2);
    /// assert_eq!(mmap.multi_len(), 3);
    /// ```
    #[inline]
    pub fn push(&mut self, value: V) {
        self.inner.get_mut().push(value);
        self.inc_num_values(1);
    }

    /// Tries to remove the value at `index` from this occupied entry, shifting the remaining values, if any.
    /// On success, returns the [`RemoveResult`].
    /// Returns this entry back if `index` is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use minimultimap::{ MultiMap, Entry };
    ///
    /// let mut mmap = MultiMap::from([(7, "foo".to_string()), (9, "bar".to_string()), (7, "baz".to_string())]);
    /// assert_eq!(mmap.len(), 2);
    /// assert_eq!(mmap.multi_len(), 3);
    ///
    /// match mmap.entry(7) {
    ///     Entry::Occupied(mut entry) => {
    ///         match entry.remove_at(2) {
    ///             Ok(_) => unreachable!("index `2` should be out of bounds"),
    ///             Err(entry) => {
    ///                 let removed = entry.remove_at(0).unwrap();
    ///                 assert_eq!(removed.removed, "foo".to_string());
    ///                 let remaining = removed.remaining.unwrap();
    ///                 assert_eq!(remaining.get().len(), 1);
    ///             },
    ///         }
    ///     },
    ///     Entry::Vacant(_) => unreachable!(),
    /// }
    ///
    /// assert_eq!(mmap[&7], ["baz".to_string()]);
    /// assert_eq!(mmap.len(), 2);
    /// assert_eq!(mmap.multi_len(), 2);
    /// ```
    #[inline]
    pub fn remove_at(self, index: usize) -> Result<RemoveResult<'a, K, V>, Self> {
        self.remove_at_impl(index, false)
    }

    /// Tries to remove the value at `index` from this occupied entry, swapping with the last value, if any.
    /// On success, returns the [`RemoveResult`].
    /// Returns this entry back if `index` is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use minimultimap::{ MultiMap, Entry };
    ///
    /// let mut mmap = MultiMap::from([(7, "foo".to_string()), (9, "bar".to_string()), (7, "baz".to_string()), (7, "bob".to_string())]);
    /// assert_eq!(mmap.len(), 2);
    /// assert_eq!(mmap.multi_len(), 4);
    ///
    /// match mmap.entry(7) {
    ///     Entry::Occupied(mut entry) => {
    ///         match entry.swap_remove_at(3) {
    ///             Ok(_) => unreachable!("index `3` should be out of bounds"),
    ///             Err(entry) => {
    ///                 let removed = entry.swap_remove_at(0).unwrap();
    ///                 assert_eq!(removed.removed, "foo".to_string());
    ///                 let remaining = removed.remaining.unwrap();
    ///                 assert_eq!(remaining.get().len(), 2);
    ///             },
    ///         }
    ///     },
    ///     Entry::Vacant(_) => unreachable!(),
    /// }
    ///
    /// assert_eq!(mmap[&7], ["bob".to_string(), "baz".to_string()]);
    /// assert_eq!(mmap.len(), 2);
    /// assert_eq!(mmap.multi_len(), 3);
    /// ```
    #[inline]
    pub fn swap_remove_at(self, index: usize) -> Result<RemoveResult<'a, K, V>, Self> {
        self.remove_at_impl(index, true)
    }

    /// Pops the last value from this entry.
    ///
    /// # Examples
    ///
    /// ```
    /// use minimultimap::{ MultiMap, Entry, OccupiedEntry };
    ///
    /// let mut mmap = MultiMap::from([(7, "foo".to_string()), (9, "bar".to_string()), (7, "baz".to_string())]);
    /// assert_eq!(mmap.len(), 2);
    /// assert_eq!(mmap.multi_len(), 3);
    ///
    /// match mmap.entry(7) {
    ///     Entry::Occupied(entry) => {
    ///         let removed = entry.pop();
    ///         assert_eq!(removed.removed, "baz".to_string());
    ///         let remaining = removed.remaining.unwrap();
    ///         assert_eq!(remaining.get().len(), 1);
    ///     },
    ///     Entry::Vacant(_) => unreachable!(),
    /// }
    ///
    /// assert_eq!(mmap[&7], ["foo".to_string()]);
    /// assert_eq!(mmap.len(), 2);
    /// assert_eq!(mmap.multi_len(), 2);
    /// ```
    #[inline]
    pub fn pop(self) -> RemoveResult<'a, K, V> {
        let len = self.get().len();
        debug_assert!(len > 0);
        unsafe {
            self.remove_at(len - 1)
                .unwrap_unchecked_dbg_msg("`pop()` should never fail")
        }
    }

    #[inline]
    fn get_entry_values(&self) -> &EntryValues<V> {
        self.inner.get()
    }

    fn remove_at_impl(mut self, index: usize, swap: bool) -> Result<RemoveResult<'a, K, V>, Self> {
        // Temporarily swap the existing value(s) with a dummy.
        let empty = EntryValues::Multiple(vec![]);
        let previous = std::mem::replace(self.inner.get_mut(), empty);
        match previous.remove_impl(index, swap) {
            Ok(EntryValuesRemoveResult { removed, remaining }) => {
                if let Some(remaining) = remaining {
                    // Removed the non-last value. Put the remaining value(s) back, return the removed value and the entry.
                    let empty_ = std::mem::replace(self.inner.get_mut(), remaining);
                    debug_assert!(matches!(empty_, EntryValues::Multiple(..)));
                    debug_assert!(empty_.is_empty());
                    Self::dec_num_values(&mut self.num_values, 1);
                    Ok(RemoveResult::not_last(removed, self))
                } else {
                    // Removed the last value. Remove the entry, return the removed value.
                    let empty_ = self.inner.remove();
                    debug_assert!(matches!(empty_, EntryValues::Multiple(..)));
                    debug_assert!(empty_.is_empty());
                    Self::dec_num_values(&mut self.num_values, 1);
                    Ok(RemoveResult::last(removed))
                }
            }
            Err(values) => {
                // Index out of bounds. Put the value(s) back, return the entry back.
                let empty_ = std::mem::replace(self.inner.get_mut(), values);
                debug_assert!(matches!(empty_, EntryValues::Multiple(..)));
                debug_assert!(empty_.is_empty());
                Err(self)
            }
        }
    }

    fn inc_num_values(&mut self, num: usize) {
        debug_assert!(num > 0);
        unsafe { *self.num_values.as_mut() += num };
    }

    // `num` can be `0`.
    fn dec_num_values(num_values: &mut NonNull<usize>, num: usize) {
        let current_num_values = unsafe { *num_values.as_ref() };
        debug_assert!(current_num_values >= num);
        unsafe { *num_values.as_mut() -= num };
    }
}

impl<'a, K, V> OccupiedEntry<'a, K, V>
where
    V: PartialEq,
{
    /// Tries to insert the (unique) `value` at `index` into this entry.
    /// Returns `None` on success, else returns the error.
    ///
    /// NOTE: value uniqueness is checked by linearly scanning all values in the entry, which might be suboptimal.
    ///
    /// # Examples
    ///
    /// ```
    /// use minimultimap::{ MultiMap, Entry, InsertError, InsertErrorKind };
    /// use std::num::NonZeroUsize;
    ///
    /// let mut mmap = MultiMap::from([(7, "foo".to_string()), (9, "bar".to_string())]);
    /// assert_eq!(mmap.len(), 2);
    /// assert_eq!(mmap.multi_len(), 2);
    ///
    /// match mmap.entry(7) {
    ///     Entry::Occupied(mut entry) => {
    ///         assert_eq!(
    ///             entry.insert_at_unique(2, "baz".to_string()),
    ///             Some(InsertError{ value: "baz".to_string(), error: InsertErrorKind::IndexOutOfBounds(NonZeroUsize::new(1).unwrap()) })
    ///         );
    ///         assert_eq!(
    ///             entry.insert_at_unique(0, "foo".to_string()),
    ///             Some(InsertError{ value: "foo".to_string(), error: InsertErrorKind::ValueNotUnique })
    ///         );
    ///         assert!(entry.insert_at_unique(0, "baz".to_string()).is_none());
    ///     },
    ///     Entry::Vacant(_) => {},
    /// }
    ///
    /// assert_eq!(mmap[&7], ["baz".to_string(), "foo".to_string()]);
    /// assert_eq!(mmap.len(), 2);
    /// assert_eq!(mmap.multi_len(), 3);
    /// ```
    #[inline]
    pub fn insert_at_unique(&mut self, index: usize, value: V) -> Option<InsertError<V>> {
        self.inner
            .get_mut()
            .insert_unique(index, value)
            .or_else(|| {
                self.inc_num_values(1);
                None
            })
    }

    /// Pushes the (unique) `value` to the end of this entry.
    /// Returns `None` on success, else returns the `value` back if it is not unique.
    ///
    /// NOTE: value uniqueness is checked by linearly scanning all values in the entry, which might be suboptimal.
    ///
    /// # Examples
    ///
    /// ```
    /// use minimultimap::{ MultiMap, Entry };
    ///
    /// let mut mmap = MultiMap::from([(7, "foo".to_string()), (9, "bar".to_string())]);
    /// assert_eq!(mmap.len(), 2);
    /// assert_eq!(mmap.multi_len(), 2);
    ///
    /// match mmap.entry(7) {
    ///     Entry::Occupied(mut entry) => {
    ///         assert_eq!(entry.push_unique("foo".to_string()).unwrap(), "foo".to_string());
    ///         assert!(entry.push_unique("baz".to_string()).is_none());
    ///     },
    ///     Entry::Vacant(_) => {},
    /// }
    ///
    /// assert_eq!(mmap[&7], ["foo".to_string(), "baz".to_string()]);
    /// assert_eq!(mmap.len(), 2);
    /// assert_eq!(mmap.multi_len(), 3);
    /// ```
    #[inline]
    pub fn push_unique(&mut self, value: V) -> Option<V> {
        self.inner.get_mut().push_unique(value).or_else(|| {
            self.inc_num_values(1);
            None
        })
    }
}

impl<'a, K: 'a, V: 'a> VacantEntry<'a, K, V> {
    /// See [`std::collections::hash_map::VacantEntry::key`].
    #[inline]
    pub fn key(&self) -> &K {
        self.inner.key()
    }

    /// See [`std::collections::hash_map::VacantEntry::into_key`].
    #[inline]
    pub fn into_key(self) -> K {
        self.inner.into_key()
    }

    /// See [`std::collections::hash_map::VacantEntry::insert`].
    ///
    /// # Examples
    ///
    /// ```
    /// use minimultimap::{ MultiMap, Entry, EntryValues };
    ///
    /// let mut mmap = MultiMap::from([(7, "foo".to_string()), (9, "bar".to_string())]);
    /// assert_eq!(mmap.len(), 2);
    /// assert_eq!(mmap.multi_len(), 2);
    ///
    /// match mmap.entry(11) {
    ///     Entry::Occupied(_) => unreachable!(),
    ///     Entry::Vacant(entry) => {
    ///         let inserted = entry.insert("baz".to_string());
    ///         assert_eq!(inserted, &"baz".to_string());
    ///     },
    /// }
    ///
    /// assert_eq!(mmap[&11], ["baz".to_string()]);
    /// assert_eq!(mmap.len(), 3);
    /// assert_eq!(mmap.multi_len(), 3);
    /// ```
    #[inline]
    pub fn insert(mut self, value: V) -> &'a mut V {
        let inserted = self.inner.insert(EntryValues::One(value));
        unsafe { *self.num_values.as_mut() += 1 };
        debug_assert_eq!(inserted.len(), 1);
        &mut inserted[0]
    }
}

impl<K, V, S> FromIterator<(K, V)> for MultiMap<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher + Default,
{
    /// See [`HashMap::from_iter()`](HashMap::from_iter).
    ///
    /// NOTE: allows non-unique values per key.
    /// You may use [`MultiMap::from_iter_unique`] to only keep unique values.
    ///
    /// # Examples
    ///
    /// ```
    /// use minimultimap::MultiMap;
    ///
    /// let mmap = MultiMap::<i32, String>::from_iter([(7, "foo".to_string()), (9, "bar".to_string()), (7, "foo".to_string()), (9, "baz".to_string())]);
    ///
    /// assert_eq!(mmap.len(), 2);
    /// assert_eq!(mmap.multi_len(), 4);
    ///
    /// assert_eq!(mmap[&7], ["foo".to_string(), "foo".to_string()]);
    /// assert_eq!(mmap[&9], ["bar".to_string(), "baz".to_string()]);
    /// ```
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        let mut map = MultiMap::with_hasher(Default::default());
        map.extend(iter);
        map
    }
}

impl<'a, K, V, S> FromIterator<(&'a K, &'a V)> for MultiMap<K, V, S>
where
    K: Eq + Hash + Copy,
    V: Copy,
    S: BuildHasher + Default,
{
    /// See [`HashMap::from_iter()`](HashMap::from_iter).
    ///
    /// NOTE: allows non-unique values per key.
    /// You may use [`from_iter_unique()`](MultiMap::from_iter_unique) to only keep unique values.
    ///
    /// # Examples
    ///
    /// ```
    /// use minimultimap::MultiMap;
    ///
    /// let vec: Vec<_> = [(7, true), (9, false), (7, false), (9, true), (7, true)].into();
    /// let mmap = MultiMap::<i32, bool>::from_iter(vec);
    ///
    /// assert_eq!(mmap.len(), 2);
    /// assert_eq!(mmap.multi_len(), 5);
    ///
    /// assert_eq!(mmap[&7], [true, false, true]);
    /// assert_eq!(mmap[&9], [false, true]);
    /// ```
    fn from_iter<T: IntoIterator<Item = (&'a K, &'a V)>>(iter: T) -> Self {
        let mut map = MultiMap::with_hasher(Default::default());
        map.extend(iter);
        map
    }
}

/// Like [`FromIterator`], but implementations must only accept unique items (i.e. those which don't already exist in the collection),
/// while skipping non-unique.
pub trait FromIteratorUnique<A>: FromIterator<A>
where
    A: PartialEq,
{
    /// See [`FromIterator::from_iter`], but implementations must only accept unique items (i.e. those which don't already exist in the collection),
    /// while skipping non-unique.
    fn from_iter_unique<T: IntoIterator<Item = A>>(iter: T) -> Self;
}

impl<K, V, S> FromIteratorUnique<(K, V)> for MultiMap<K, V, S>
where
    K: Eq + Hash,
    V: PartialEq,
    S: BuildHasher + Default,
{
    /// See [`HashMap::from_iter()`](HashMap::from_iter), but only accepts unique values per key.
    ///
    /// NOTE: value uniqueness is checked by linearly scanning all values associated with each key, which might be suboptimal.
    ///
    /// You may use [`MultiMap::from_iter`] to allow non-unique values.
    ///
    /// # Examples
    ///
    /// ```
    /// use minimultimap::{ MultiMap, FromIteratorUnique };
    ///
    /// let mmap = MultiMap::<i32, String>::from_iter_unique([(7, "foo".to_string()), (9, "bar".to_string()), (7, "foo".to_string()), (9, "baz".to_string())]);
    ///
    /// assert_eq!(mmap.len(), 2);
    /// assert_eq!(mmap.multi_len(), 3);
    ///
    /// assert_eq!(mmap[&7], ["foo".to_string()]);
    /// assert_eq!(mmap[&9], ["bar".to_string(), "baz".to_string()]);
    /// ```
    fn from_iter_unique<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        let mut map = MultiMap::with_hasher(Default::default());
        map.extend_unique(iter);
        map
    }
}

impl<'a, K, V, S> FromIteratorUnique<(&'a K, &'a V)> for MultiMap<K, V, S>
where
    K: Eq + Hash + Copy,
    V: PartialEq + Copy,
    S: BuildHasher + Default,
{
    /// See [`HashMap::from_iter()`](HashMap::from_iter), but only accepts unique values per key.
    ///
    /// NOTE: value uniqueness is checked by linearly scanning all values associated with each key, which might be suboptimal.
    ///
    /// You may use [`MultiMap::from_iter`] to allow non-unique values.
    ///
    /// # Examples
    ///
    /// ```
    /// use minimultimap::{ MultiMap, FromIteratorUnique };
    ///
    /// let vec: Vec<_> = [(7, true), (9, false), (7, false), (9, true), (7, true)].into();
    /// let mmap = MultiMap::<i32, bool>::from_iter_unique(vec.iter().map(|t| (&t.0, &t.1)));
    ///
    /// assert_eq!(mmap.len(), 2);
    /// assert_eq!(mmap.multi_len(), 4);
    ///
    /// assert_eq!(mmap[&7], [true, false]);
    /// assert_eq!(mmap[&9], [false, true]);
    /// ```
    fn from_iter_unique<T: IntoIterator<Item = (&'a K, &'a V)>>(iter: T) -> Self {
        let mut map = MultiMap::with_hasher(Default::default());
        map.extend_unique(iter);
        map
    }
}

impl<K, V, S> Extend<(K, V)> for MultiMap<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    /// See [`HashMap::extend()`](HashMap::extend).
    ///
    /// NOTE: allows non-unique values per key.
    /// You may use [`extend_unique()`](ExtendUnique::extend_unique) to only keep unique values.
    ///
    /// # Examples
    ///
    /// ```
    /// use minimultimap::MultiMap;
    ///
    /// let mut mmap = MultiMap::new();
    /// mmap.extend([(7, "foo".to_string()), (9, "bar".to_string()), (7, "foo".to_string()), (9, "baz".to_string())]);
    ///
    /// assert_eq!(mmap.len(), 2);
    /// assert_eq!(mmap.multi_len(), 4);
    ///
    /// assert_eq!(mmap[&7], ["foo".to_string(), "foo".to_string()]);
    /// assert_eq!(mmap[&9], ["bar".to_string(), "baz".to_string()]);
    /// ```
    #[inline]
    fn extend<T: IntoIterator<Item = (K, V)>>(&mut self, iter: T) {
        for (key, value) in iter {
            self.add(key, value);
        }
    }
}

impl<'a, K, V, S> Extend<(&'a K, &'a V)> for MultiMap<K, V, S>
where
    K: Eq + Hash + Copy,
    V: Copy,
    S: BuildHasher,
{
    /// See [`HashMap::extend()`](HashMap::extend), for copyable keys and values.
    ///
    /// NOTE: allows non-unique values per key.
    /// You may use [`extend_unique()`](ExtendUnique::extend_unique) to only keep unique values.
    ///
    /// # Examples
    ///
    /// ```
    /// use minimultimap::MultiMap;
    ///
    /// let mut mmap = MultiMap::<i32, bool>::new();
    /// let vec: Vec<_> = [(7, true), (9, false), (7, false), (9, true), (7, true)].into();
    /// mmap.extend(vec.iter().map(|t| (&t.0, &t.1)));
    ///
    /// assert_eq!(mmap.len(), 2);
    /// assert_eq!(mmap.multi_len(), 5);
    ///
    /// assert_eq!(mmap[&7], [true, false, true]);
    /// assert_eq!(mmap[&9], [false, true]);
    /// ```
    #[inline]
    fn extend<T: IntoIterator<Item = (&'a K, &'a V)>>(&mut self, iter: T) {
        for (key, value) in iter {
            self.add(*key, *value);
        }
    }
}

/// Like [`Extend`], but implementations must only accept unique items (i.e. those which don't already exist in the collection),
/// while skipping non-unique.
pub trait ExtendUnique<A>: Extend<A> {
    /// See [`Extend::extend`], but implementations must only accept unique items (i.e. those which don't already exist in the collection),
    /// while skipping non-unique.
    fn extend_unique<T: IntoIterator<Item = A>>(&mut self, iter: T);
}

impl<K, V, S> ExtendUnique<(K, V)> for MultiMap<K, V, S>
where
    K: Eq + Hash,
    V: PartialEq,
    S: BuildHasher,
{
    /// See [`HashMap::extend()`](HashMap::extend), but only accepts unique values per key.
    ///
    /// NOTE: value uniqueness is checked by linearly scanning all values associated with each key, which might be suboptimal.
    ///
    /// # Examples
    ///
    /// ```
    /// use minimultimap::{MultiMap, ExtendUnique};
    ///
    /// let mut mmap = MultiMap::new();
    /// mmap.extend_unique([(7, "foo".to_string()), (9, "bar".to_string()), (7, "foo".to_string()), (9, "baz".to_string())]);
    ///
    /// assert_eq!(mmap.len(), 2);
    /// assert_eq!(mmap.multi_len(), 3);
    ///
    /// assert_eq!(mmap[&7], ["foo".to_string()]);
    /// assert_eq!(mmap[&9], ["bar".to_string(), "baz".to_string()]);
    /// ```
    #[inline]
    fn extend_unique<T: IntoIterator<Item = (K, V)>>(&mut self, iter: T) {
        for (key, value) in iter {
            self.add_unique(key, value);
        }
    }
}

impl<'a, K, V, S> ExtendUnique<(&'a K, &'a V)> for MultiMap<K, V, S>
where
    K: Eq + Hash + Copy,
    V: PartialEq + Copy,
    S: BuildHasher,
{
    /// See [`HashMap::extend()`](HashMap::extend), for copyable keys and values, but only accepts unique values per key.
    ///
    /// NOTE: value uniqueness is checked by linearly scanning all values associated with each key, which might be suboptimal.
    ///
    /// # Examples
    ///
    /// ```
    /// use minimultimap::{MultiMap, ExtendUnique};
    ///
    /// let mut mmap = MultiMap::<i32, bool>::new();
    /// let vec: Vec<_> = [(7, true), (9, false), (7, false), (9, true), (7, true)].into();
    /// mmap.extend_unique(vec.iter().map(|t| (&t.0, &t.1)));
    ///
    /// assert_eq!(mmap.len(), 2);
    /// assert_eq!(mmap.multi_len(), 4);
    ///
    /// assert_eq!(mmap[&7], [true, false]);
    /// assert_eq!(mmap[&9], [false, true]);
    /// ```
    #[inline]
    fn extend_unique<T: IntoIterator<Item = (&'a K, &'a V)>>(&mut self, iter: T) {
        for (key, value) in iter {
            self.add_unique(*key, *value);
        }
    }
}

/// Iterator over (references to) all key-value tuples in the [`MultiMap`], in unspecified order.
///
/// Similar to [`Iter`], but the same key may be returned multiple times with values which
/// might or might not be unique, depending on how they were inserted into the [`MultiMap`].
///
/// This struct is created by the [`multi_iter()`](MultiMap::multi_iter) method on [`MultiMap`]. See its documentation for more.
pub struct MultiIter<'a, K, V> {
    /// Iterates over hashmap keys.
    map_iter: Option<Iter<'a, K, EntryValues<V>>>,
    /// Iterates over elements of the `Entry::Multiple()` entry.
    entry_iter: Option<(&'a K, SliceIter<'a, V>)>,
    /// Number of remaining values in the iterator.
    num_values: usize,
}

impl<'a, K, V> MultiIter<'a, K, V> {
    fn new(map_iter: Iter<'a, K, EntryValues<V>>, num_values: usize) -> Self {
        Self {
            map_iter: Some(map_iter),
            entry_iter: None,
            num_values,
        }
    }
}

impl<'a, K, V> Iterator for MultiIter<'a, K, V> {
    type Item = (&'a K, &'a V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // First finish iterating over the current `Entry::Multiple()` entry, if any.
        if let Some((k, entry_iter)) = self.entry_iter.as_mut() {
            if let Some(v) = entry_iter.next() {
                debug_assert!(self.num_values > 0);
                self.num_values -= 1;
                return Some((*k, v));
            } else {
                // Don't forget to clear the entry iterator when done with it before moving on to the next key.
                self.entry_iter.take();
            }
        }

        // Otherwise go to the next hashmap key.
        if let Some((k, v)) = self.map_iter.as_mut().and_then(Iterator::next) {
            debug_assert!(self.num_values > 0);
            self.num_values -= 1;
            match v {
                EntryValues::One(v) => Some((k, v)),
                EntryValues::Multiple(values) => {
                    debug_assert!(values.len() >= 2);
                    let mut values = values.iter();
                    let v = unsafe {
                        values
                            .next()
                            .unwrap_unchecked_dbg_msg("we know `values` contains at least 2 values")
                    };
                    self.entry_iter.replace((k, values));
                    Some((k, v))
                }
            }
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.num_values, Some(self.num_values))
    }
}

impl<'a, K, V> ExactSizeIterator for MultiIter<'a, K, V> {
    #[inline]
    fn len(&self) -> usize {
        self.num_values
    }
}

impl<'a, K, V> FusedIterator for MultiIter<'a, K, V> {}

/// Iterator over (mutable references to) all key-value tuples in the [`MultiMap`], in unspecified order.
///
/// Similar to [`Iter`], but the same key may be returned multiple times with values which
/// might or might not be unique, depending on how they were inserted into the [`MultiMap`].
///
/// This struct is created by the [`multi_iter_mut()`](MultiMap::multi_iter_mut) method on [`MultiMap`]. See its documentation for more.
pub struct MultiIterMut<'a, K, V> {
    /// Iterates over hashmap keys.
    map_iter: Option<IterMut<'a, K, EntryValues<V>>>,
    /// Iterates over elements of the `Entry::Multiple()` entry.
    entry_iter: Option<(&'a K, SliceIterMut<'a, V>)>,
    /// Number of remaining values in the iterator.
    num_values: usize,
}

impl<'a, K, V> MultiIterMut<'a, K, V> {
    fn new(map_iter: IterMut<'a, K, EntryValues<V>>, num_values: usize) -> Self {
        Self {
            map_iter: Some(map_iter),
            entry_iter: None,
            num_values,
        }
    }
}

impl<'a, K, V> Iterator for MultiIterMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // First finish iterating over the current `Entry::Multiple()` entry, if any.
        if let Some((k, entry_iter)) = self.entry_iter.as_mut() {
            if let Some(v) = entry_iter.next() {
                debug_assert!(self.num_values > 0);
                self.num_values -= 1;
                return Some((*k, v));
            } else {
                // Don't forget to clear the entry iterator when done with it before moving on to the next key.
                self.entry_iter.take();
            }
        }

        // Otherwise go to the next hashmap key.
        if let Some((k, v)) = self.map_iter.as_mut().and_then(Iterator::next) {
            debug_assert!(self.num_values > 0);
            self.num_values -= 1;
            match v {
                EntryValues::One(v) => Some((k, v)),
                EntryValues::Multiple(values) => {
                    debug_assert!(values.len() >= 2);
                    let mut values = values.iter_mut();
                    let v = unsafe {
                        values
                            .next()
                            .unwrap_unchecked_dbg_msg("we know `values` contains at least 2 values")
                    };
                    self.entry_iter.replace((k, values));
                    Some((k, v))
                }
            }
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.num_values, Some(self.num_values))
    }
}

impl<'a, K, V> ExactSizeIterator for MultiIterMut<'a, K, V> {
    #[inline]
    fn len(&self) -> usize {
        self.num_values
    }
}

impl<'a, K, V> FusedIterator for MultiIterMut<'a, K, V> {}

/// `Vec::insert`, but without a panic - the inserted value is returned back if `index` is out of bounds.
fn vec_insert<T>(vec: &mut Vec<T>, index: usize, element: T) -> Option<T> {
    let len = vec.len();

    if index > len {
        return Some(element);
    }

    // space for the new element
    if len == vec.capacity() {
        vec.reserve(1);
    }

    unsafe {
        // infallible
        // The spot to put the new value
        {
            let p = vec.as_mut_ptr().add(index);
            #[allow(clippy::comparison_chain)]
            if index < len {
                // Shift everything over to make space. (Duplicating the
                // `index`th element into two consecutive places.)
                std::ptr::copy(p, p.add(1), len - index);
            } else {
                debug_assert_eq!(index, len);
                // No elements need shifting.
            }
            // Write it in, overwriting the first copy of the `index`th
            // element.
            std::ptr::write(p, element);
        }
        vec.set_len(len + 1);
    }

    None
}

/// `Vec::remove`, but without a panic - returns `None` if `index` is out of bounds.
fn vec_remove<T>(vec: &mut Vec<T>, index: usize) -> Option<T> {
    let len = vec.len();
    (index < len).then(|| unsafe {
        // infallible
        let ret;
        {
            // the place we are taking from.
            let ptr = vec.as_mut_ptr().add(index);
            // copy it out, unsafely having a copy of the value on
            // the stack and in the vector at the same time.
            ret = std::ptr::read(ptr);

            // Shift everything down to fill in that spot.
            std::ptr::copy(ptr.add(1), ptr, len - index - 1);
        }
        vec.set_len(len - 1);
        ret
    })
}

/// `Vec::swap_remove`, but without a panic - returns `None` if `index` is out of bounds.
fn vec_swap_remove<T>(vec: &mut Vec<T>, index: usize) -> Option<T> {
    let len = vec.len();
    (index < len).then(||
        // We replace self[index] with the last element. Note that if the
        // bounds check above succeeds there must be a last element (which
        // can be self[index] itself).
        unsafe {
            let value = std::ptr::read(vec.as_ptr().add(index));
            let base_ptr = vec.as_mut_ptr();
            std::ptr::copy(base_ptr.add(len - 1), base_ptr.add(index), 1);
            vec.set_len(len - 1);
            value
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vec_insert_test() {
        let v = vec![7, 8];
        {
            let mut v = v.clone();
            assert!(vec_insert(&mut v, 0, 9).is_none());
            assert_eq!(v, [9, 7, 8]);
        }
        {
            let mut v = v.clone();
            assert!(vec_insert(&mut v, 1, 9).is_none());
            assert_eq!(v, [7, 9, 8]);
        }
        {
            let mut v = v.clone();
            assert!(vec_insert(&mut v, 2, 9).is_none());
            assert_eq!(v, [7, 8, 9]);
        }
        {
            let mut v = v.clone();
            assert_eq!(vec_insert(&mut v, 3, 9), Some(9));
        }
        {
            let mut v: Vec<i32> = vec![];
            assert!(vec_insert(&mut v, 0, 7).is_none());
            assert_eq!(v, [7]);
        }
    }

    #[test]
    fn vec_remove_test() {
        let v = vec![7, 8, 9];
        {
            let mut v = v.clone();
            assert_eq!(vec_remove(&mut v, 0), Some(7));
            assert_eq!(v, [8, 9]);
        }
        {
            let mut v = v.clone();
            assert_eq!(vec_remove(&mut v, 1), Some(8));
            assert_eq!(v, [7, 9]);
        }
        {
            let mut v = v.clone();
            assert_eq!(vec_remove(&mut v, 2), Some(9));
            assert_eq!(v, [7, 8]);
        }
        {
            let mut v = v.clone();
            assert!(vec_remove(&mut v, 3).is_none());
        }
        {
            let mut v: Vec<i32> = vec![];
            assert!(vec_remove(&mut v, 0).is_none());
            assert!(vec_remove(&mut v, 1).is_none());
        }
    }

    #[test]
    fn vec_swap_remove_test() {
        let v = vec![7, 8, 9];
        {
            let mut v = v.clone();
            assert_eq!(vec_swap_remove(&mut v, 0), Some(7));
            assert_eq!(v, [9, 8]);
        }
        {
            let mut v = v.clone();
            assert_eq!(vec_swap_remove(&mut v, 1), Some(8));
            assert_eq!(v, [7, 9]);
        }
        {
            let mut v = v.clone();
            assert_eq!(vec_swap_remove(&mut v, 2), Some(9));
            assert_eq!(v, [7, 8]);
        }
        {
            let mut v = v.clone();
            assert!(vec_swap_remove(&mut v, 3).is_none());
        }
        {
            let mut v: Vec<i32> = vec![];
            assert!(vec_swap_remove(&mut v, 0).is_none());
            assert!(vec_swap_remove(&mut v, 1).is_none());
        }
    }

    #[test]
    fn values_insert_non_unique_test() {
        {
            let v = EntryValues::One(7);
            {
                let mut v = v.clone();
                assert!(v.insert(0, 9).is_none());
                assert_eq!(v, EntryValues::Multiple(vec![9, 7]));
            }
            {
                let mut v = v.clone();
                assert!(v.insert(1, 9).is_none());
                assert_eq!(v, EntryValues::Multiple(vec![7, 9]));
            }
            {
                let mut v = v.clone();
                assert!(v.insert(0, 7).is_none());
                assert_eq!(v, EntryValues::Multiple(vec![7, 7]));
            }
            {
                let mut v = v.clone();
                assert_eq!(v.insert(2, 9), Some(9));
            }
        }
        {
            let v = EntryValues::Multiple(vec![7, 8]);
            {
                let mut v = v.clone();
                assert!(v.insert(0, 9).is_none());
                assert_eq!(v, EntryValues::Multiple(vec![9, 7, 8]));
            }
            {
                let mut v = v.clone();
                assert!(v.insert(1, 9).is_none());
                assert_eq!(v, EntryValues::Multiple(vec![7, 9, 8]));
            }
            {
                let mut v = v.clone();
                assert!(v.insert(2, 9).is_none());
                assert_eq!(v, EntryValues::Multiple(vec![7, 8, 9]));
            }

            {
                let mut v = v.clone();
                assert!(v.insert(0, 7).is_none());
                assert_eq!(v, EntryValues::Multiple(vec![7, 7, 8]));
            }
            {
                let mut v = v.clone();
                assert!(v.insert(1, 7).is_none());
                assert_eq!(v, EntryValues::Multiple(vec![7, 7, 8]));
            }
            {
                let mut v = v.clone();
                assert_eq!(v.insert(3, 9), Some(9));
            }
        }
    }

    #[test]
    fn values_insert_unique_test() {
        {
            let v = EntryValues::One(7);
            {
                let mut v = v.clone();
                assert!(v.insert_unique(0, 9).is_none());
                assert_eq!(v, EntryValues::Multiple(vec![9, 7]));
            }
            {
                let mut v = v.clone();
                assert!(v.insert_unique(1, 9).is_none());
                assert_eq!(v, EntryValues::Multiple(vec![7, 9]));
            }
            {
                let mut v = v.clone();
                assert_eq!(
                    v.insert_unique(0, 7).unwrap(),
                    InsertError {
                        value: 7,
                        error: InsertErrorKind::ValueNotUnique
                    }
                );
            }
            {
                let mut v = v.clone();
                assert_eq!(
                    v.insert_unique(2, 7).unwrap(),
                    InsertError {
                        value: 7,
                        error: InsertErrorKind::IndexOutOfBounds(NonZeroUsize::new(1).unwrap())
                    }
                );
            }
        }
        {
            let v = EntryValues::Multiple(vec![7, 8]);
            {
                let mut v = v.clone();
                assert!(v.insert_unique(0, 9).is_none());
                assert_eq!(v, EntryValues::Multiple(vec![9, 7, 8]));
            }
            {
                let mut v = v.clone();
                assert!(v.insert_unique(1, 9).is_none());
                assert_eq!(v, EntryValues::Multiple(vec![7, 9, 8]));
            }
            {
                let mut v = v.clone();
                assert!(v.insert_unique(2, 9).is_none());
                assert_eq!(v, EntryValues::Multiple(vec![7, 8, 9]));
            }
            {
                let mut v = v.clone();
                assert_eq!(
                    v.insert_unique(0, 7).unwrap(),
                    InsertError {
                        value: 7,
                        error: InsertErrorKind::ValueNotUnique
                    }
                );
            }
            {
                let mut v = v.clone();
                assert_eq!(
                    v.insert_unique(3, 7).unwrap(),
                    InsertError {
                        value: 7,
                        error: InsertErrorKind::IndexOutOfBounds(NonZeroUsize::new(2).unwrap())
                    }
                );
            }
        }
    }
}
