//! # minimultimap
//!
//! A simple minimalistic multimap wrapper around a [`HashMap`],
//! optimized for the usual case of having one value per key with no overhead
//! (i.e. a `Vec` is not allocated to hold a single value),
//! but which does support multiple values by storing them in a `Vec` when necessary.
//!
//! Implements only a limited necessary subset of [`HashMap`] functionality.

use std::{
    borrow::Borrow,
    collections::{
        hash_map::{Entry as HashMapEntry, Iter as HashMapIter, OccupiedEntry, RandomState},
        HashMap,
    },
    hash::{BuildHasher, Hash},
    hint::unreachable_unchecked,
    iter::Iterator,
    num::NonZeroUsize,
    ops::{Deref, DerefMut},
    slice::{self, Iter as SliceIter, IterMut as SliceIterMut},
};

/// One or multiple values in the multimap associated with a given key.
///
/// Derefs to a slice.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Entry<T> {
    /// One value associated with a key.
    One(T),
    /// Multiple values associated with a key.
    Multiple(Vec<T>),
}

impl<T> Deref for Entry<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        match self {
            Entry::One(val) => slice::from_ref(val),
            Entry::Multiple(values) => values,
        }
    }
}

impl<T> DerefMut for Entry<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Entry::One(val) => slice::from_mut(val),
            Entry::Multiple(values) => values,
        }
    }
}

/// Mutable reference to the existing entry in the multimap
/// for one or multiple values associated with a given key.
///
/// Allows mutating the entry in the multimap by inserting or removing values.
///
/// Derefs to a slice.
pub struct EntryMut<'a, K, V>(OccupiedEntry<'a, K, Entry<V>>);

impl<'a, K, V> Deref for EntryMut<'a, K, V> {
    type Target = [V];

    fn deref(&self) -> &Self::Target {
        self.0.get()
    }
}

impl<'a, K, V> DerefMut for EntryMut<'a, K, V> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.get_mut()
    }
}

impl<'a, K, V> EntryMut<'a, K, V> {
    /// Inserts another `value` into this entry.
    ///
    /// Returns the (non-zero) number of values associated with the `key` before this call.
    ///
    /// NOTE: order of values in the [`EntryMut`], if there are multiple, is unspecified.
    pub fn insert(&mut self, value: V) -> NonZeroUsize {
        insert_into_entry(&mut self.0, value)
    }

    /// Removes the value from this entry at given `index`.
    ///
    /// NOTE: order of values in the [`EntryMut`], if there are multiple, is unspecified.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    pub fn remove(mut self, index: usize) -> V {
        match self.0.get_mut() {
            Entry::One(_) => {
                assert_eq!(index, 0);
                match self.0.remove_entry().1 {
                    Entry::One(v) => v,
                    // We know the current entry is for a single value.
                    Entry::Multiple(_) => unsafe { unreachable_unchecked() },
                }
            }
            Entry::Multiple(values) => {
                debug_assert!(values.len() >= 2);
                let val = values.swap_remove(index);
                if values.len() == 1 {
                    let remaining = match values.pop() {
                        Some(remaining) => remaining,
                        // We know there's exactly one element in the `Vec`.
                        None => unsafe { unreachable_unchecked() },
                    };
                    let _ = std::mem::replace(self.0.get_mut(), Entry::One(remaining));
                }
                val
            }
        }
    }

    /// Removes the value from the entry at given `index` without checking its validity.
    ///
    /// NOTE: order of values in the [`EntryMut`], if there are multiple, is unspecified.
    ///
    /// # Safety
    ///
    /// `index` must not be out of bounds.
    pub unsafe fn remove_unchecked(mut self, index: usize) -> V {
        match self.0.get_mut() {
            Entry::One(_) => {
                debug_assert_eq!(index, 0);
                match self.0.remove_entry().1 {
                    Entry::One(v) => v,
                    // We know the current entry is for a single value.
                    Entry::Multiple(_) => unreachable_unchecked(),
                }
            }
            Entry::Multiple(values) => {
                debug_assert!(values.len() >= 2);
                let val = swap_remove(values, index);
                if values.len() == 1 {
                    let remaining = values
                        .pop()
                        // We know there's exactly one element in the `Vec`.
                        .unwrap_or_else(|| unsafe { unreachable_unchecked() });

                    let _ = std::mem::replace(self.0.get_mut(), Entry::One(remaining));
                }
                val
            }
        }
    }
}

/// `Vec::swap_remove`, but without a panic.
/// The caller guarantess `index` is strictly less than `vec.len()`.
unsafe fn swap_remove<T>(vec: &mut Vec<T>, index: usize) -> T {
    let len = vec.len();
    debug_assert!(index < len);

    // We replace self[index] with the last element. Note that if the
    // bounds check above succeeds there must be a last element (which
    // can be self[index] itself).
    let value = std::ptr::read(vec.as_ptr().add(index));
    let base_ptr = vec.as_mut_ptr();
    std::ptr::copy(base_ptr.add(len - 1), base_ptr.add(index), 1);
    vec.set_len(len - 1);
    value
}

/// A simple multimap wrapper around a [`HashMap`],
/// optimized for the usual case of having one value per key with no overhead
/// (i.e. a `Vec` is not allocated to hold a single value),
/// but which does support multiple values by storing them in a `Vec` when necessary.
///
/// Implements only a limited necessary subset of [`HashMap`] functionality.
pub struct MultiMap<K, V, S = RandomState>(HashMap<K, Entry<V>, S>);

impl<K, V> MultiMap<K, V, RandomState> {
    pub fn new() -> Self {
        Self(HashMap::new())
    }
}

impl<K, V, S> MultiMap<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    /// See [`HashMap::insert`].
    ///
    /// Returns the number of values associated with the `key` before this call.
    /// `0` means a new multimap entry was created for `key`.
    pub fn insert(&mut self, key: K, value: V) -> usize {
        match self.0.entry(key) {
            HashMapEntry::Occupied(mut entry) => insert_into_entry(&mut entry, value).get(),
            // No values for this key - insert a single value.
            HashMapEntry::Vacant(entry) => {
                entry.insert(Entry::One(value));
                0
            }
        }
    }

    /// See [`HashMap::remove`].
    ///
    /// Returns [`all`](Entry) values associated with the `key`, if there are any, one or multiple.
    pub fn remove<Q: ?Sized>(&mut self, key: &Q) -> Option<Entry<V>>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.0.remove_entry(key.borrow()).map(|(_, value)| value)
    }

    /// See [`HashMap::get`].
    ///
    /// Returns a reference to a slice of values in the multimap associated with the `key`, if there are any, one or multiple.
    ///
    /// NOTE: order of values in the slice, if there are multiple, is unspecified.
    pub fn get<Q: ?Sized>(&self, key: &Q) -> Option<&[V]>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.get_entry(key).map(Entry::deref)
    }

    /// See [`HashMap::get_mut`].
    ///
    /// Returns a mutable reference to a slice of values in the multimap associated with the `key`, if there are any, one or multiple.
    ///
    /// NOTE: order of values in the slice, if there are multiple, is unspecified.
    pub fn get_mut<Q: ?Sized>(&mut self, key: &Q) -> Option<&mut [V]>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.get_entry_mut(key).map(Entry::deref_mut)
    }

    /// See [`HashMap::entry`], but only returns a mutable reference to an existing entry, if any.
    ///
    /// Allows to remove a single value associated with the `key`, see [`EntryMut::remove`].
    pub fn entry(&mut self, key: K) -> Option<EntryMut<'_, K, V>> {
        match self.0.entry(key) {
            HashMapEntry::Occupied(entry) => Some(EntryMut(entry)),
            HashMapEntry::Vacant(_) => None,
        }
    }

    /// Returns an iterator over values in the multimap associated with `key`, if there are any, in unspecified order.
    pub fn get_iter<Q: ?Sized>(&self, key: &Q) -> impl Iterator<Item = &V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        EntryIter(self.get(key.borrow()).map(<[V]>::iter))
    }

    /// Returns an iterator over values in the multimap associated with `key`, if there are any, in unspecified order.
    pub fn get_iter_mut<Q: ?Sized>(&mut self, key: &Q) -> impl Iterator<Item = &mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        EntryIterMut(self.get_mut(key.borrow()).map(<[V]>::iter_mut))
    }

    /// See [`HashMap::clear`].
    pub fn clear(&mut self) {
        self.0.clear()
    }

    /// See [`HashMap::values`].
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.0.values().map(Entry::deref).flat_map(<[V]>::iter)
    }

    /// See [`HashMap::iter`].
    pub fn iter(&self) -> impl Iterator<Item = (&K, &Entry<V>)> {
        self.0.iter()
    }

    /// Returns an iterator over all key-value tuples in the multimap, in unspecified order.
    /// The same key may be returned more than once with different values.
    pub fn multi_iter(&self) -> impl Iterator<Item = (&K, &V)> {
        MultiMapIter::new(self.0.iter())
    }

    fn get_entry<Q: ?Sized>(&self, key: &Q) -> Option<&Entry<V>>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.0.get(key.borrow())
    }

    fn get_entry_mut<Q: ?Sized>(&mut self, key: &Q) -> Option<&mut Entry<V>>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.0.get_mut(key.borrow())
    }
}

struct EntryIter<'a, V>(Option<SliceIter<'a, V>>);

impl<'a, V> Iterator for EntryIter<'a, V> {
    type Item = &'a V;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.as_mut().and_then(SliceIter::next)
    }
}

struct EntryIterMut<'a, V>(Option<SliceIterMut<'a, V>>);

impl<'a, V> Iterator for EntryIterMut<'a, V> {
    type Item = &'a mut V;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.as_mut().and_then(SliceIterMut::next)
    }
}

struct MultiMapIter<'a, K, V> {
    // Iterates over hashmap keys.
    map_iter: Option<HashMapIter<'a, K, Entry<V>>>,
    // Iterates over elements of the `Entry::Multiple()` entry.
    entry_iter: Option<(&'a K, SliceIter<'a, V>)>,
}

impl<'a, K, V> MultiMapIter<'a, K, V> {
    fn new(map_iter: HashMapIter<'a, K, Entry<V>>) -> Self {
        Self {
            map_iter: Some(map_iter),
            entry_iter: None,
        }
    }
}

impl<'a, K, V> Iterator for MultiMapIter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        // First finish iterating over the current `Entry::Multiple()` entry, if any.
        if let Some((k, entry_iter)) = self.entry_iter.as_mut() {
            if let Some(v) = entry_iter.next() {
                return Some((*k, v));
            } else {
                // Don't forget to clear the entry iterator when done with it before moving on to the next key.
                self.entry_iter.take();
            }
        }

        // Otherwise go to the next hashmap key.
        if let Some((k, v)) = self.map_iter.as_mut().and_then(Iterator::next) {
            match v {
                Entry::One(v) => Some((k, v)),
                Entry::Multiple(values) => {
                    debug_assert!(values.len() >= 2);
                    let mut values = values.iter();
                    let v = values
                        .next()
                        // We know `values` contains at least 2 values.
                        .unwrap_or_else(|| unsafe { unreachable_unchecked() });
                    self.entry_iter.replace((k, values));
                    Some((k, v))
                }
            }
        } else {
            None
        }
    }
}

fn insert_into_entry<'a, K, V>(
    entry: &mut OccupiedEntry<'a, K, Entry<V>>,
    value: V,
) -> NonZeroUsize {
    match entry.get_mut() {
        // A single value for this key - transform into `Multiple`.
        Entry::One(_) => {
            // Preallocate a `Vec` for 2 values; push the new value.
            let value = {
                let mut vec = Vec::with_capacity(2);
                vec.push(value);
                vec
            };
            // Replace the existing single value with a `Vec` containing the new value,
            // and then push the existing value to the vec.
            // This avoids removing/re-inserting the hashmap entry.
            // TODO: is there a way to do it without `unsafe`?
            let existing = std::mem::replace(entry.get_mut(), Entry::Multiple(value));
            if let Entry::Multiple(value) = entry.get_mut() {
                debug_assert!(value.len() == 1);
                if let Entry::One(existing) = existing {
                    // For now pushing to the front to keep order, it's cheap with only one copy.
                    // Alternatively could just `push()` - simpler,
                    // but order of first two values per key is swapped.
                    //value.push(existing);
                    vec_push_front(value, existing);
                } else {
                    // We know the existing value was `One`.
                    unsafe { unreachable_unchecked() };
                }
            } else {
                // We know the just inserted value was `Multiple`.
                unsafe { unreachable_unchecked() };
            }
            unsafe { NonZeroUsize::new_unchecked(1) }
        }
        // `Multiple` values for this key - just append to the vec.
        Entry::Multiple(existing) => {
            let len = existing.len();
            debug_assert!(len >= 2);
            existing.push(value);
            unsafe { NonZeroUsize::new_unchecked(len) }
        }
    }
}

/// See `Vec::insert()`, but
/// - with `index == 0`,
/// - assumes a pre-reserved `vec` with enough capacity for one more `value`.
fn vec_push_front<T>(vec: &mut Vec<T>, value: T) {
    debug_assert!(vec.capacity() > vec.len());

    let len = vec.len();

    unsafe {
        // infallible
        // The spot to put the new value
        {
            let p = vec.as_mut_ptr();
            // Shift everything over to make space. (Duplicating the
            // `index`th element into two consecutive places.)
            std::ptr::copy(p, p.offset(1), len);
            // Write it in, overwriting the first copy of the `index`th
            // element.
            std::ptr::write(p, value);
        }
        vec.set_len(len + 1);
    }
}

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn multimap() {
        let mut map = MultiMap::<u32, String>::new();

        assert!(map.get(&0).is_none());

        assert_eq!(map.insert(0, "foo".into()), 0);
        assert_eq!(map.get(&0).unwrap(), ["foo".to_string()]);
        assert!(map.get_iter(&0).eq(["foo".to_string()].iter()));
        assert!(map.get(&1).is_none());
        assert!(map
            .multi_iter()
            .eq([(0u32, "foo".to_string())].iter().map(|(k, v)| (k, v))));

        let foo = map.get_mut(&0).unwrap();
        foo[0] = "bar".into();

        assert_eq!(map.get(&0).unwrap(), ["bar".to_string()]);
        assert!(map.get_iter(&0).eq(["bar".to_string()].iter()));
        assert!(map.get(&1).is_none());
        assert!(map
            .multi_iter()
            .eq([(0u32, "bar".to_string())].iter().map(|(k, v)| (k, v))));

        assert_eq!(map.insert(1, "foo".into()), 0);
        assert_eq!(map.get(&0).unwrap(), ["bar".to_string()]);
        assert!(map.get_iter(&0).eq(["bar".to_string()].iter()));
        assert_eq!(map.get(&1).unwrap(), ["foo".to_string()]);
        assert!(map.get_iter(&1).eq(["foo".to_string()].iter()));

        for (k, v) in map.multi_iter() {
            if *k == 0 {
                assert_eq!(v, "bar");
            }
            if *k == 1 {
                assert_eq!(v, "foo");
            }
        }

        assert_eq!(map.insert(0, "baz".into()), 1);
        assert!(
            map.get(&0).unwrap() == ["bar".to_string(), "baz".to_string()]
                || map.get(&0).unwrap() == ["baz".to_string(), "bar".to_string()]
        );
        for v in map.get_iter(&0) {
            assert!(v == "bar" || v == "baz");
        }
        assert_eq!(map.get(&1).unwrap(), ["foo".to_string()]);
        assert!(map.get_iter(&1).eq(["foo".to_string()].iter()));

        for (k, v) in map.multi_iter() {
            if *k == 0 {
                assert!(v == "bar" || v == "baz");
            }
            if *k == 1 {
                assert_eq!(v, "foo");
            }
        }

        map.get_iter_mut(&0)
            .enumerate()
            .for_each(|(idx, value)| value.push_str(&idx.to_string()));
        assert!(
            map.get(&0).unwrap() == ["bar0".to_string(), "baz1".to_string()]
                || map.get(&0).unwrap() == ["baz0".to_string(), "bar1".to_string()]
        );
        for v in map.get_iter(&0) {
            assert!(v == "bar0" || v == "baz1" || v == "bar1" || v == "baz0");
        }
        assert_eq!(map.get(&1).unwrap(), ["foo".to_string()]);
        assert!(map.get_iter(&1).eq(["foo".to_string()].iter()));

        for (k, v) in map.multi_iter() {
            if *k == 0 {
                assert!(v == "bar0" || v == "baz1" || v == "bar1" || v == "baz0");
            }
            if *k == 1 {
                assert_eq!(v, "foo");
            }
        }

        assert_eq!(map.insert(0, "bob2".into()), 2);
        assert!(
            map.get(&0).unwrap() == ["bar0".to_string(), "baz1".to_string(), "bob2".to_string()]
                || map.get(&0).unwrap()
                    == ["bar0".to_string(), "bob2".to_string(), "baz1".to_string()]
                || map.get(&0).unwrap()
                    == ["bob2".to_string(), "bar0".to_string(), "baz1".to_string()]
                || map.get(&0).unwrap()
                    == ["baz1".to_string(), "bar0".to_string(), "bob2".to_string()]
                || map.get(&0).unwrap()
                    == ["baz1".to_string(), "bob2".to_string(), "bar0".to_string()]
                || map.get(&0).unwrap()
                    == ["bob2".to_string(), "baz1".to_string(), "bar0".to_string()]
                || map.get(&0).unwrap()
                    == ["bar1".to_string(), "baz0".to_string(), "bob2".to_string()]
                || map.get(&0).unwrap()
                    == ["bar1".to_string(), "bob2".to_string(), "baz0".to_string()]
                || map.get(&0).unwrap()
                    == ["bob2".to_string(), "bar1".to_string(), "baz0".to_string()]
                || map.get(&0).unwrap()
                    == ["baz0".to_string(), "bar1".to_string(), "bob2".to_string()]
                || map.get(&0).unwrap()
                    == ["baz0".to_string(), "bob2".to_string(), "bar1".to_string()]
                || map.get(&0).unwrap()
                    == ["bob2".to_string(), "baz0".to_string(), "bar1".to_string()]
        );
        for v in map.get_iter(&0) {
            assert!(v == "bar0" || v == "baz1" || v == "bar1" || v == "baz0" || v == "bob2");
        }
        assert_eq!(map.get(&1).unwrap(), ["foo".to_string()]);
        assert!(map.get_iter(&1).eq(["foo".to_string()].iter()));

        for (k, v) in map.multi_iter() {
            if *k == 0 {
                assert!(v == "bar0" || v == "baz1" || v == "bar1" || v == "baz0" || v == "bob2");
            }
            if *k == 1 {
                assert_eq!(v, "foo");
            }
        }

        let bar_baz_bob = map.remove(&0).unwrap();
        assert_eq!(bar_baz_bob.len(), 3);
        assert!(
            bar_baz_bob.contains(&"bar0".to_string()) || bar_baz_bob.contains(&"bar1".to_string())
        );
        assert!(
            bar_baz_bob.contains(&"baz0".to_string()) || bar_baz_bob.contains(&"baz1".to_string())
        );
        assert!(bar_baz_bob.contains(&"bob2".to_string()));

        assert!(map.get(&0).is_none());
        assert_eq!(map.get(&1).unwrap(), ["foo".to_string()]);
        assert!(map.get_iter(&1).eq(["foo".to_string()].iter()));

        assert!(map
            .multi_iter()
            .eq([(1u32, "foo".to_string())].iter().map(|(k, v)| (k, v))));

        assert_eq!(map.remove(&1).unwrap(), Entry::One("foo".to_string(),));

        assert!(map.get(&0).is_none());
        assert!(map.get(&1).is_none());

        assert_eq!(map.insert(2, "bill".into()), 0);

        assert!(map.get(&0).is_none());
        assert!(map.get(&1).is_none());
        assert_eq!(map.get(&2).unwrap(), ["bill".to_string()]);
        assert!(map.get_iter(&2).eq(["bill".to_string()].iter()));

        assert!(map
            .multi_iter()
            .eq([(2u32, "bill".to_string())].iter().map(|(k, v)| (k, v))));

        assert_eq!(map.insert(2, "charles".into()), 1);

        assert!(map.get(&0).is_none());
        assert!(map.get(&1).is_none());
        assert!(
            map.get(&2).unwrap() == ["bill".to_string(), "charles".to_string()]
                || map.get(&2).unwrap() == ["charles".to_string(), "bill".to_string()]
        );
        for v in map.get_iter(&0) {
            assert!(v == "bill" || v == "charles");
        }

        for (k, v) in map.multi_iter() {
            if *k == 2 {
                assert!(v == "bill" || v == "charles");
            }
        }

        let bill_charles = map.entry(2).unwrap();
        assert_eq!(bill_charles.len(), 2);
        let bill_or_charles = bill_charles.remove(0);
        assert!(bill_or_charles == "bill".to_string() || bill_or_charles == "charles".to_string());

        assert!(map.get(&0).is_none());
        assert!(map.get(&1).is_none());
        assert!(
            map.get(&2).unwrap() == ["bill".to_string()]
                || map.get(&2).unwrap() == ["charles".to_string()]
        );
        assert!(
            map.get_iter(&2).eq(["charles".to_string()].iter())
                || map.get_iter(&2).eq(["bill".to_string()].iter())
        );

        let mut bill_or_charles = map.entry(2).unwrap();
        assert_eq!(bill_or_charles.len(), 1);

        assert_eq!(bill_or_charles.insert("dog".to_string()).get(), 1);

        assert!(
            map.get(&2).unwrap() == ["dog".to_string(), "charles".to_string()]
                || map.get(&2).unwrap() == ["charles".to_string(), "dog".to_string()]
                || map.get(&2).unwrap() == ["dog".to_string(), "bill".to_string()]
                || map.get(&2).unwrap() == ["bill".to_string(), "dog".to_string()]
        );
        for v in map.get_iter(&0) {
            assert!(v == "dog" || v == "charles" || v == "bill");
        }

        for (k, v) in map.multi_iter() {
            if *k == 2 {
                assert!(v == "dog" || v == "charles" || v == "bill");
            }
        }

        let dog_and_bill_or_charles = map.entry(2).unwrap();
        let dog_or_bill_or_charles = dog_and_bill_or_charles.remove(1);
        assert!(
            dog_or_bill_or_charles == "dog".to_string()
                || dog_or_bill_or_charles == "bill".to_string()
                || dog_or_bill_or_charles == "charles".to_string()
        );
        let dog_or_bill_or_charles = map.entry(2).unwrap();
        let dog_or_bill_or_charles = dog_or_bill_or_charles.remove(0);
        assert!(
            dog_or_bill_or_charles == "dog".to_string()
                || dog_or_bill_or_charles == "bill".to_string()
                || dog_or_bill_or_charles == "charles".to_string()
        );

        assert!(map.get(&0).is_none());
        assert!(map.get(&1).is_none());
        assert!(map.get(&2).is_none());
    }
}
