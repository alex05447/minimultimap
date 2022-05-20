# minimultimap

A simple minimalistic multimap wrapper around a [`HashMap`],
optimized for the usual case of having one value per key with no overhead
(i.e. a `Vec` is not allocated to hold a single value),
but which does support multiple values by storing them in a `Vec` when necessary.

Implements only a limited necessary subset of [`HashMap`] functionality.

[`HashMap`]: https://doc.rust-lang.org/1.60.0/std/collections/hash_map/struct.HashMap.html