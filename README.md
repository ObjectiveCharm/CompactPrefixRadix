# Introduction
A very compact implementation of Radix tree in Rust with extended feature in prefix match.
It is originally designed for a network project, but I think it is a good idea to extract it as a standalone library.
I forced AI to write the code.

# Usage
It provides a KV store like generic interface, with some extended feature in prefix match.
```rust
    use compact_radix::RadixTree;
    // Since we use a compile-time determined path length, you need to specify it in creation
    let mut tree = RadixTree::<_, 64>::new();
    // insert a key-value pair, the old value will be returned if the key already exists 
    let old_value  = tree.insert("test", String::from("test_val"));
    // longest prefix match
    let longest_val = tree.longest_prefix_match("testing123");
    // first (shortest) prefix match
    let first_val =  tree.first_prefix_match("testing123");
    // exact match
    let exact_val = tree.exact_match("test");
    // remove a key-value pair
    let removed_val = tree.remove("test");
    // get the number of key-value pairs
    let len = tree.len();
    // drain all key-value pairs
    tree.clean();
```

# Notice
- The storage is designed for light-wight CoW value which can be cloned cheaply.
- If you want to store heavy value, you should use `Arc` or `Rc` to wrap it, or struggle with the borrow checker and lifetime.
- It is compatible with ascii characters, it is trade-off, add support to complete Unicode characters will introduce overhead, and the performance of bitmap will drastically decrease.
  But in order to support Unicode characters, you can convert it to binary blob and specify a large enough path length, and it will work.
- It uses u64 size, in 32 bit build, software emulation will be used, and that will result in a performance penalty.


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