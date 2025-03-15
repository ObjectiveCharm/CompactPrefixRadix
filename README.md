# Introduction
A very compact implementation of Radix tree in Rust with extended feature in prefix match.
It is originally designed for a network project, but I think it is a good idea to extract it as a standalone library.
No third party dependency is used, and it is pure Rust code.
I forced AI to help me to write the code.
__!!!If you are using v0.1.0, please update, the old version have a memory leak problem because of my carelessness (I forgot to remove extraneous unused Box::leak() product!!!__

# Usag
creation, deletion and manipulate
```rust
    use compact_radix::RadixTree;
    // Since we use a compile-time determined path length, you need to specify it in creation
    let mut tree = RadixTree::<_, 64>::new();
    // insert a key-value pair, the old value will be returned if the key already exists 
    let old_value  = tree.insert("test", String::from("test_val"));
    // remove a key-value pair
    let removed_val = tree.remove("test");
    // get the number of key-value pairs
    let len = tree.len();
    // drain all key-value pairs
    tree.clean();
```

For queries of Clone types, a CoW style is preferred, and you can use the following interface to access the values by key
```rust
    // longest prefix match
    let longest_val = tree.longest_prefix_match("testing123");
    // first (shortest) prefix match
    let first_val =  tree.first_prefix_match("testing123");
    // exact match
    let exact_val = tree.exact_match("test");
```

For non Clone types, we offer interface to get the value by reference, both mutable and immutable.
But in concurrent scenarios, to avoid complex borrow checker and lifetime struggling, mechanism such as `Arc` and `Mutex` may be better
```rust
    /// Retrieve a reference to the value associated with the input key's pattern.
/// Returns the shortest prefix match along with its length, if found.
/// **Notice: exact matched input won't be considered as a prefix match**
pub fn first_prefix_match_ref<'a>(&'a self, input: &str) -> Option<(&'a V, usize)> {...}

/// Retrieve a mutable reference to the value associated with the input key's pattern.
/// Returns the shortest prefix match along with its length, if found.
/// **Notice: exact matched input won't be considered as a prefix match**
pub fn first_prefix_match_mut<'a>(&'a mut self, input: &str) -> Option<(&'a mut V, usize)> {...}

/// Retrieve all references to values associated with the input key's patterns,
/// in ascending order of length.
pub fn all_prefix_matches_ref<'a>(&'a self, input: &str) -> Vec<(&'a V, usize)> {...}

/// Retrieve all mutable references to values associated with the input key's patterns,
/// in ascending order of length.
pub fn all_prefix_matches_mut<'a>(&'a mut self, input: &str) -> Vec<(&'a mut V, usize)> {...}

/// Retrieve a reference to the value associated with the longest prefix match for the input key.
pub fn longest_prefix_match_ref<'a>(&'a self, input: &str) -> Option<(&'a V, usize)> {...}
/// Retrieve a mutable reference to the value associated with the longest prefix match for the input key.
pub fn longest_prefix_match_mut<'a>(&'a mut self, input: &str) -> Option<(&'a mut V, usize)> {...}

/// Retrieve a reference to the exact match value for the input key, if it exists.
pub fn exact_match_ref<'a>(&'a self, input: &str) -> Option<&'a V> {...}

/// Retrieve a mutable reference to the exact match value for the input key, if it exists.
pub fn exact_match_mut<'a>(&'a mut self, input: &str) -> Option<&'a mut V> {...}
```


# Notice
- It is compatible with ascii characters, it is trade-off, add support to complete Unicode characters will introduce overhead, and the performance of bitmap will drastically decrease.
  But in order to support Unicode characters, you can convert it to binary blob and specify a large enough path length, and it will work.
- It uses u64 size, in 32 bit build, software emulation will be used, and that will result in a performance penalty.

# TODO
- Iterator for matching
- Iterator for all entries
- Concurrent support (Idk. I thought there can be some concurrent b+ tree like structure with more locking granularity, but it will be a lot of work, and may be not necessary, for concurrent b+ tree is mature enough, fully verified and widely used, the later one also has better locality and cache affinity)

# Efficiency
The implementation tries best to be both memory and time efficient.
1. Fixed length bitmap is used to store the indices of children, rather than hashmap or vector.
The path is stored in a fixed length array, rather than a heap allocated string.
But as a trade-off, the path length is fixed at compile time and cannot be scaled at runtime.
There is some space overhead for each node, but it is very small.We also provide some preset constant for path length, you can choose the one that fits your need.
2. We used radix tree, which is more efficient than trie in terms of memory usage.
3. We used raw pointer carefully to avoid unnecessary smart point or borrow checker overhead.
4. If you want to get more cutting-edge performance, please see [blart](https://github.com/declanvk/blart), it is [Adaptive Radix Tree](https://db.in.tum.de/~leis/papers/ART.pdf) based and even more efficient.

```rust
// Preset path length
pub const DOMAIN_PATH_LEN: usize = 256;
pub const IPV4_PATH_LEN: usize = 32;
pub const IPV6_PATH_LEN: usize = 128;
```

It may introduce some overhead for small path length, but it is still acceptable (it is a small array with char), and aligned in memory.
And there is assertions in it, if the path length is too long, it will panic at runtime.
In the future I may implement a simd-based batch operation, I have no idea.