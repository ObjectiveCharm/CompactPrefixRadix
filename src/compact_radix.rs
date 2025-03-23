use std::ptr::NonNull;
use crate::helper::{char_to_index, CHAR_SET_SIZE};

struct Node<V, const N: usize> {
    edge: [char; N],
    edge_len: usize,
    value: Option<V>,
    // 改为使用固定大小的位图数组,支持128位
    children_mask: [u64; 2],
    children: [Option<NonNull<Node<V, N>>>; CHAR_SET_SIZE],
    parent: Option<ParentRef<V, N>>,
}

/// Refers to the parent node, containing a pointer to the parent node and the index in the parent node
struct ParentRef<V, const N: usize> {
    node: NonNull<Node<V, N>>,
    index: usize,
}

pub struct RadixTree<V, const N: usize> {
    root: NonNull<Node<V, N>>,
    len: usize,
}

impl<V, const N: usize> Node<V, N> {
    fn new() -> Box<Self> {
        Box::new(Node {
            edge: ['\0'; N],
            edge_len: 0,
            value: None,
            children_mask: [0; 2], // 初始化为全0
            children: std::array::from_fn(|_| None),
            parent: None,
        })
    }

    #[inline]
    fn has_child(&self, index: usize) -> bool {
        let array_idx = index / 64;  // 确定在哪个u64中
        let bit_idx = index % 64;    // 确定在u64中的位置
        (self.children_mask[array_idx] & (1 << bit_idx)) != 0
    }

    #[inline]
    fn set_child(&mut self, index: usize) {

        let array_idx = index / 64;
        let bit_idx = index % 64;
        self.children_mask[array_idx] |= 1 << bit_idx;
    }

    #[inline]
    fn clear_child(&mut self, index: usize) {
        let array_idx = index / 64;
        let bit_idx = index % 64;
        self.children_mask[array_idx] &= !(1 << bit_idx);
    }

    fn count_children(&self) -> u32 {
        self.children_mask[0].count_ones() + self.children_mask[1].count_ones()
    }

    fn get_only_child(&self) -> Option<NonNull<Node<V, N>>> {
        // 计算两个u64中1的总数
        let ones_count = self.children_mask[0].count_ones() +
            self.children_mask[1].count_ones();

        if ones_count != 1 {
            return None;
        }

        // 找到置1的位置
        if self.children_mask[0] != 0 {
            let idx = self.children_mask[0].trailing_zeros() as usize;
            self.children[idx]
        } else {
            let idx = self.children_mask[1].trailing_zeros() as usize + 64;
            self.children[idx]
        }
    }

    fn can_merge_with_child(&self) -> bool {
        self.value.is_none() &&
            self.count_children() == 1 &&
            self.edge_len + self.get_only_child().map_or(0, |child| unsafe {
                (*child.as_ref()).edge_len
            }) <= N
    }

    fn common_prefix_len(&self, key: &str) -> usize {
        self.edge[..self.edge_len]
            .iter()
            .zip(key.chars())
            .take_while(|(a, b)| *a == b)
            .count()
    }

    fn with_edge(edge_str: &str) -> Option<Box<Self>> {
        if edge_str.len() > N {
            return None;
        }

        let mut node = Self::new();
        // for (i, c) in edge_str.chars().enumerate() {
        //     node.edge[i] = c;
        // }
        // node.edge_len = edge_str.len();
        node.set_edge(edge_str); // 这里会进行断言检查
        Some(node)
    }

    // 生成edge时的断言
    fn set_edge(&mut self, s: &str) {
        assert!(s.len() <= N,
                "Edge length {} exceeds maximum allowed length {}",
                s.len(), N);
        self.edge.fill('\0');
        for (i, c) in s.chars().enumerate() {
            self.edge[i] = c;
        }
        self.edge_len = s.len();
    }

    fn edge_str(&self) -> String {
        self.edge[..self.edge_len].iter().collect()
    }

    unsafe fn split(this: *mut Self, split_pos: usize) -> Option<Box<Self>> {
        let this_ref = &mut *this;
        if split_pos >= this_ref.edge_len || split_pos >= N {
            return None;
        }

        // 创建新节点
        let mut new_node = Self::new();

        // 设置新节点的edge（前半部分）
        new_node.edge[..split_pos].copy_from_slice(&this_ref.edge[..split_pos]);
        new_node.edge_len = split_pos;

        // 更新原节点（后半部分）
        let remaining_edge: Vec<_> = this_ref.edge[split_pos..this_ref.edge_len].iter().cloned().collect();
        this_ref.edge.fill('\0');
        for (i, &c) in remaining_edge.iter().enumerate() {
            this_ref.edge[i] = c;
        }
        this_ref.edge_len = remaining_edge.len();

        // // 特殊情况：如果原节点有值且分割点正好是原edge的长度
        // let original_value = if split_pos == this_ref.edge_len {
        //     this_ref.value.take()  // 将值移到新节点
        // } else {
        //     None
        // };


        // 设置父子关系
        if let Some(&first_char) = remaining_edge.first() {
            if let Some(idx) = char_to_index(first_char) {
                // 先获取新节点的裸指针
                let new_node_ptr = Box::into_raw(new_node);

                // 设置父子关系
                (*new_node_ptr).children[idx] = Some(NonNull::new(this).unwrap());
                (*new_node_ptr).set_child(idx);
                this_ref.parent = Some(ParentRef {
                    node: NonNull::new(new_node_ptr).unwrap(),
                    index: idx,
                });

                // 将新节点转回Box返回
                return Some(unsafe { Box::from_raw(new_node_ptr) });
            }
        }

        //// 如果有原始值，恢复到新节点
        // new_node.value = original_value;
        Some(new_node)
    }

}

impl<V, const N: usize> RadixTree<V, N> {
    /// Creation
    pub fn new() -> Self {
        RadixTree {
            root: NonNull::new(Box::into_raw(Node::new())).unwrap(),
            len: 0,
        }
    }

    /// Insertion
    /// When there is an existing value associated with the input key, it will be replaced by the new value and returned
    pub fn insert(&mut self, key: &str, value: V) -> Option<V> {
        if key.is_empty() {
            // 空key直接插入根节点
            unsafe {
                let old_value = (*self.root.as_ptr()).value.replace(value);
                if old_value.is_none() {
                    self.len += 1;
                }
                return old_value;
            }
        }

        unsafe {
            self.insert_internal(self.root.as_ptr(), key, value, None)
        }
    }

    /// Removal
    /// Return the value associated with the input key if it exists
    pub fn remove(&mut self, key: &str) -> Option<V> {
        if key.is_empty() {
            return None;
        }

        unsafe {
            let node = self.find_node(key)?;
            self.remove_node(node)
        }
    }

    /// Number of elements
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the structure is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Clean / drain the structure
    pub fn clear(&mut self) {
        unsafe {
            drop(Box::from_raw(self.root.as_ptr()));
            self.root = NonNull::new(Box::into_raw(Node::new())).unwrap();
            self.len = 0;
        }
    }
}
impl<V: Clone, const N: usize> RadixTree<V, N> {
    /// Exact match
    pub fn exact_match(&self, input: &str) -> Option<V> {
        unsafe {
            self.find_first_match(input, true)
                .and_then(|(node, _)| (*node).value.as_ref().cloned())
        }
    }

    /// Retrieve the value associated with the input key's pattern, which is stored in the structure.
    /// That means the pattern (as the key) is the shortest prefix of the input
    /// This operation will stop as soon as the first match (but could be not the longest) is found
    /// **Notice: exact matched input won't be considered as a prefix match**
    /// For example I think "test" is not a prefix match of "test", but "test1" is
    pub fn first_prefix_match(&self, input: &str) -> Option<(V, usize)> {
        unsafe {
            self.find_first_match(input, false)
                .map(|(node, len)| ((*node).value.as_ref().unwrap().clone(), len))
        }
    }

    /// Retrieve all values associated with the input key's pattern, which is stored in the structure.
    /// That means the keys are matched pattern, in ascending order of length
    pub fn all_prefix_matches(&self, input: &str) -> Vec<(V, usize)> {
        unsafe {
            if let Some((first_node, first_len)) = self.find_first_match(input, false) {
                // 在这里将引用转换为所有权类型
                self.find_all_matches_from(first_node, first_len, input)
                    .into_iter()
                    .map(|(v, len)| (v.clone(), len))
                    .collect()
            } else {
                Vec::new()
            }
        }
    }

    /// Retrieve the value associated with the input key's pattern, which is stored in the structure.
    /// That means the pattern (as the key) is the longest prefix of the input
    /// This operation will continue searching after the first match until the end of the input string
    // It reused the logic of all_prefix_matches
    pub fn longest_prefix_match(&self, input: &str) -> Option<(V, usize)> {
        self.all_prefix_matches(input).into_iter().last()
    }
}

// Get by immutable/mutable reference
// It works for value type that doesn't implement Clone
impl<V, const N: usize> RadixTree<V, N> {
    /// Retrieve a reference to the value associated with the input key's pattern.
    /// Returns the shortest prefix match along with its length, if found.
    /// **Notice: exact matched input won't be considered as a prefix match**
    pub fn first_prefix_match_ref<'a>(&'a self, input: &str) -> Option<(&'a V, usize)> {
        unsafe {
            self.find_first_match(input, false)
                .map(|(node, len)| ((*node).value.as_ref().unwrap(), len))
        }
    }

    /// Retrieve a mutable reference to the value associated with the input key's pattern.
    /// Returns the shortest prefix match along with its length, if found.
    /// **Notice: exact matched input won't be considered as a prefix match**
    pub fn first_prefix_match_mut<'a>(&'a mut self, input: &str) -> Option<(&'a mut V, usize)> {
        unsafe {
            // We need to convert the const pointer to mut
            if let Some((node, len)) = self.find_first_match(input, false) {
                let node_mut = node as *mut Node<V, N>;
                Some(((*node_mut).value.as_mut().unwrap(), len))
            } else {
                None
            }
        }
    }

    /// Retrieve all references to values associated with the input key's patterns,
    /// in ascending order of length.
    pub fn all_prefix_matches_ref<'a>(&'a self, input: &str) -> Vec<(&'a V, usize)> {
        unsafe {
            if let Some((first_node, first_len)) = self.find_first_match(input, false) {
                self.find_all_matches_from(first_node, first_len, input)
            } else {
                Vec::new()
            }
        }
    }

    /// Retrieve all mutable references to values associated with the input key's patterns,
    /// in ascending order of length.
    pub fn all_prefix_matches_mut<'a>(&'a mut self, input: &str) -> Vec<(&'a mut V, usize)> {
        unsafe {
            if let Some((first_node, first_len)) = self.find_first_match(input, false) {
                // First collect the node pointers and match lengths
                let matches = self.collect_match_nodes(first_node, first_len, input);

                // Then convert them to mutable references
                matches.into_iter()
                    .map(|(node_ptr, len)| {
                        let node_mut = node_ptr as *mut Node<V, N>;
                        ((*node_mut).value.as_mut().unwrap(), len)
                    })
                    .collect()
            } else {
                Vec::new()
            }
        }
    }

    /// Retrieve a reference to the value associated with the longest prefix match for the input key.
    pub fn longest_prefix_match_ref<'a>(&'a self, input: &str) -> Option<(&'a V, usize)> {
        self.all_prefix_matches_ref(input).into_iter().last()
    }

    /// Retrieve a mutable reference to the value associated with the longest prefix match for the input key.
    pub fn longest_prefix_match_mut<'a>(&'a mut self, input: &str) -> Option<(&'a mut V, usize)> {
        self.all_prefix_matches_mut(input).into_iter().last()
    }

    /// Retrieve a reference to the exact match value for the input key, if it exists.
    pub fn exact_match_ref<'a>(&'a self, input: &str) -> Option<&'a V> {
        unsafe {
            self.find_first_match(input, true)
                .and_then(|(node, _)| (*node).value.as_ref())
        }
    }

    /// Retrieve a mutable reference to the exact match value for the input key, if it exists.
    pub fn exact_match_mut<'a>(&'a mut self, input: &str) -> Option<&'a mut V> {
        unsafe {
            // Convert the result to a mutable pointer since we have &mut self
            if let Some((node, _)) = self.find_first_match(input, true) {
                let node_mut = node as *mut Node<V, N>;
                (*node_mut).value.as_mut()
            } else {
                None
            }
        }
    }

    // Helper method to collect nodes and match lengths
    unsafe fn collect_match_nodes(&self, mut current: *const Node<V, N>, mut matched_len: usize, key: &str) -> Vec<(*const Node<V, N>, usize)> {
        let mut matches = vec![(current, matched_len)];

        while !current.is_null() {
            let node = &*current;
            let remaining = &key[matched_len..];

            // If input is fully consumed, discard the last match and exit
            if remaining.is_empty() {
                matches.pop();  // Remove the last match (it's an exact match)
                break;
            }

            if let Some(idx) = char_to_index(remaining.chars().next().unwrap()) {
                if node.has_child(idx) {
                    current = node.children[idx].unwrap().as_ptr();
                    let edge_str = (*current).edge_str();

                    if let Some(remaining_after_edge) = remaining.strip_prefix(edge_str.as_str()) {
                        matched_len += edge_str.len();
                        if (*current).value.is_some() {
                            // Only add match if there are remaining characters
                            if !remaining_after_edge.is_empty() {
                                matches.push((current, matched_len));
                            }
                        }
                        continue;
                    }
                }
            }
            break;
        }
        matches
    }
}

impl<V, const N: usize> RadixTree<V, N> {
    /// Collect all entries and return a vector of raw pointers to nodes
    unsafe fn collect_entries_raw(&self, node: *const Node<V, N>, prefix: String) -> Vec<(*mut Node<V, N>, String)> {
        let mut result = Vec::new();
        self.collect_entries_recursive(node as *mut Node<V, N>, prefix, &mut result);
        result
    }

    unsafe fn collect_entries_recursive(
        &self,
        node: *mut Node<V, N>,
        prefix: String,
        result: &mut Vec<(*mut Node<V, N>, String)>,
    ) {
        let node_ref = &mut *node;
        let current_prefix = prefix + &node_ref.edge_str();

        // 添加当前节点到结果中
        result.push((node, current_prefix.clone()));

        // 递归处理所有子节点
        for i in 0..CHAR_SET_SIZE {
            if node_ref.has_child(i) {
                if let Some(child) = node_ref.children[i] {
                    self.collect_entries_recursive(child.as_ptr(), current_prefix.clone(), result);
                }
            }
        }
    }

    /// Enumerate all entries in the RadixTree, returning (key, reference to value)
    pub fn entries(&self) -> Vec<(String, &V)> {
        let mut result = Vec::new();
        unsafe {
            let raw_entries = self.collect_entries_raw(self.root.as_ptr(), String::new());
            for (node, key) in raw_entries {
                if let Some(ref value) = (*node).value {
                    result.push((key, value));
                }
            }
        }
        result
    }

    /// Clean / drain the structure and dump all entries
    pub fn clear_and_dump(&mut self) -> Vec<(String, V)> {
        let mut entries = Vec::new();
        unsafe {
            let raw_entries = self.collect_entries_raw(self.root.as_ptr(), String::new());
            for (node, key) in raw_entries {
                if let Some(value) = (*node).value.take() {
                    entries.push((key, value));
                }
            }
        }
        self.clear();
        entries
    }
}

impl<V, const N: usize> RadixTree<V, N> {
    unsafe fn find_first_match(&self, key: &str, exact: bool) -> Option<(*const Node<V, N>, usize)> {
        if key.is_empty() {
            let root = self.root.as_ptr();
            if exact && (*root).value.is_some() {
                return Some((root, 0));
            }
            return None;
        }
        let mut current = self.root.as_ptr();
        let mut matched_len = 0;
        let mut remaining = key;

        while !remaining.is_empty() {
            let first_char = remaining.chars().next().unwrap();
            let idx = char_to_index(first_char)?;
            if !(*current).has_child(idx) {
                break;
            }
            current = (*current).children[idx].unwrap().as_ptr();

            // 2. 检查edge匹配
            let node = &*current;
            let node_edge = &node.edge[..node.edge_len];
            if !remaining.starts_with(&node_edge.iter().collect::<String>()) {
                break;
            }

            // 3. 更新匹配长度和剩余输入
            matched_len += node.edge_len;
            remaining = &remaining[node.edge_len..];

            // 4. 精确匹配和前缀匹配的逻辑应分开
            if node.value.is_some() {
                if exact {
                    // 精确匹配：必须完全匹配
                    if remaining.is_empty() {
                        return Some((current, matched_len));
                    }
                } else {
                    // 前缀匹配：还有剩余字符才返回
                    if !remaining.is_empty() {
                        return Some((current, matched_len));
                    }
                }
            }
        }
        None
    }

    unsafe fn find_all_matches_from(&self, mut current: *const Node<V, N>, mut matched_len: usize, key: &str) -> Vec<(&V, usize)> {
        let mut matches = vec![((*current).value.as_ref().unwrap(), matched_len)];

        while !current.is_null() {
            let node = &*current;
            let remaining = &key[matched_len..];

            // 如果输入串被完全消耗了，丢弃最后一个匹配并退出
            if remaining.is_empty() {
                matches.pop();  // 移除最后一个匹配，因为它是精确匹配
                break;
            }

            if let Some(idx) = char_to_index(remaining.chars().next().unwrap()) {
                if node.has_child(idx) {
                    current = node.children[idx].unwrap().as_ptr();
                    let edge_str = (*current).edge_str();

                    if let Some(remaining_after_edge) = remaining.strip_prefix(edge_str.as_str()) {
                        matched_len += edge_str.len();
                        if (*current).value.is_some() {
                            // 只有当还有输入时才添加匹配
                            if !remaining_after_edge.is_empty() {
                                matches.push(((*current).value.as_ref().unwrap(), matched_len));
                            }
                        }
                        continue;
                    }
                }
            }
            break;
        }
        matches
    }

    unsafe fn insert_internal(
        &mut self,
        current: *mut Node<V, N>,
        key: &str,
        value: V,
        parent: Option<ParentRef<V, N>>
    ) -> Option<V> {
        (*current).parent = parent;

        // 如果key为空，直接在当前节点设置值
        if key.is_empty() {
            let old_value = (*current).value.replace(value);
            if old_value.is_none() {
                self.len += 1;
            }
            return old_value;
        }

        let first_char = key.chars().next().unwrap();
        let idx = match char_to_index(first_char) {
            Some(i) => i,
            None => return None,
        };

        // 如果当前位置没有子节点，创建新节点
        if !(*current).has_child(idx) {
            if let Some(mut new_node) = Node::with_edge(key) {
                new_node.value = Some(value);
                new_node.parent = Some(ParentRef {
                    node: NonNull::new(current).unwrap(),
                    index: idx,
                });

                let new_node_ptr = Box::into_raw(new_node);
                (*current).children[idx] = Some(NonNull::new(new_node_ptr).unwrap());
                (*current).set_child(idx);
                self.len += 1;
            }
            return None;
        }

        // 获取子节点并计算公共前缀长度
        let child = (*current).children[idx].unwrap().as_ptr();
        let common_len = (*child).common_prefix_len(key);

        // 如果公共前缀长度小于子节点的edge长度，需要分裂
        if common_len < (*child).edge_len {
            // 从父节点移除子节点引用
            (*current).children[idx] = None;
            (*current).clear_child(idx);

            // 分裂节点
            if let Some(new_node) = Node::split(child, common_len) {
                let new_node_ptr = Box::into_raw(new_node);
                // 设置新节点的父节点关系
                (*new_node_ptr).parent = Some(ParentRef {
                    node: NonNull::new(current).unwrap(),
                    index: idx,
                });

                // 将新节点连接到父节点
                (*current).children[idx] = Some(NonNull::new(new_node_ptr).unwrap());
                (*current).set_child(idx);

                // 如果还有剩余key，创建新叶子节点
                if common_len < key.len() {
                    let remaining = &key[common_len..];
                    if let Some(mut new_leaf) = Node::with_edge(remaining) {
                        if let Some(first_idx) = char_to_index(remaining.chars().next().unwrap()) {
                            new_leaf.value = Some(value);
                            new_leaf.parent = Some(ParentRef {
                                node: NonNull::new(new_node_ptr).unwrap(),
                                index: first_idx,
                            });

                            let new_leaf_ptr = Box::into_raw(new_leaf);
                            (*new_node_ptr).children[first_idx] = Some(NonNull::new(new_leaf_ptr).unwrap());
                            (*new_node_ptr).set_child(first_idx);
                            self.len += 1;
                        }
                    }
                    return None;
                } else {
                    // 如果没有剩余key，在新节点设置值
                    let old_value = (*new_node_ptr).value.replace(value);
                    if old_value.is_none() {
                        self.len += 1;
                    }
                    return old_value;
                }
            }
            return None;
        }

        // 如果还有剩余key，继续向下插入
        if common_len < key.len() {
            self.insert_internal(
                child,
                &key[common_len..],
                value,
                Some(ParentRef {
                    node: NonNull::new(current).unwrap(),
                    index: idx,
                })
            )
        } else {
            // 如果没有剩余key，更新子节点的值
            let old_value = (*child).value.replace(value);
            if old_value.is_none() {
                self.len += 1;
            }
            old_value
        }
    }
    unsafe fn remove_node(&mut self, mut node: NonNull<Node<V, N>>) -> Option<V> {
        let old_value = (*node.as_mut()).value.take();
        if old_value.is_none() {
            return None;
        }

        self.len -= 1;

        // 如果节点没有子节点，可以直接删除
        if (*node.as_ref()).count_children() == 0 {
            self.remove_and_repair(node.as_ptr());
        }

        old_value
    }

    unsafe fn merge_nodes(
        &mut self,
        parent: *mut Node<V, N>,
        child: *mut Node<V, N>
    ) {
        let child_ref = &*child;
        let child_edge: Vec<_> = child_ref.edge[..child_ref.edge_len]
            .iter()
            .cloned()
            .collect();
        let child_value = (*child).value.take();
        // 复制整个掩码数组
        let child_mask = child_ref.children_mask;
        let child_children = child_ref.children;

        let parent_edge_len = (*parent).edge_len;
        for (i, &c) in child_edge.iter().enumerate() {
            (*parent).edge[parent_edge_len + i] = c;
        }
        let combined_len = parent_edge_len + child_edge.len();
        assert!(combined_len <= N, "Combined edge length {} exceeds maximum allowed length {}", combined_len, N);
        (*parent).edge_len = combined_len;
        // (*parent).edge_len = parent_edge_len + child_edge.len();

        // 转移所有属性,包括两个掩码
        (*parent).children_mask = child_mask;
        (*parent).children = child_children;
        (*parent).value = child_value;

        // 更新子节点的父节点引用
        for i in 0..CHAR_SET_SIZE {
            if (*parent).has_child(i) {
                if let Some(grandchild) = (*parent).children[i] {
                    (*grandchild.as_ptr()).parent = Some(ParentRef {
                        node: NonNull::new(parent).unwrap(),
                        index: i,
                    });
                }
            }
        }

        drop(Box::from_raw(child));
    }

    unsafe fn remove_and_repair(&mut self, node: *mut Node<V, N>) {
        loop {
            // 获取parent_ref的所有权
            let parent_ref = match (*node).parent.take() {
                Some(p) => p,
                None => break,
            };

            let parent = parent_ref.node.as_ptr();
            let idx = parent_ref.index;

            // 从父节点中移除当前节点的引用
            (*parent).children[idx] = None;
            (*parent).clear_child(idx);

            // 释放当前节点
            drop(Box::from_raw(node));

            // 如果父节点可以与其唯一子节点合并
            if (*parent).can_merge_with_child() {
                if let Some(child) = (*parent).get_only_child() {
                    self.merge_nodes(parent, child.as_ptr());
                    continue;
                }
            }
            break;
        }
    }

    unsafe fn find_node(&self, key: &str) -> Option<NonNull<Node<V, N>>> {
        let mut current = self.root;
        let mut remaining = key;

        while !remaining.is_empty() {
            let first_char = remaining.chars().next().unwrap();
            let idx = char_to_index(first_char)?;

            if !(*current.as_ref()).has_child(idx) {
                return None;
            }

            current = (*current.as_ref()).children[idx].unwrap();
            let node = &*current.as_ptr();

            if !remaining.starts_with(&node.edge_str()) {
                return None;
            }

            remaining = &remaining[node.edge_len..];
        }

        Some(current)
    }
}
impl<V, const N: usize> Drop for RadixTree<V, N> {
    fn drop(&mut self) {
        unsafe {
            drop(Box::from_raw(self.root.as_ptr()));
        }
    }
}

impl<V, const N: usize> Drop for Node<V, N> {
    fn drop(&mut self) {
        // 检查两个掩码是否都为0
        if self.children_mask[0] == 0 && self.children_mask[1] == 0 {
            return;
        }

        for i in 0..CHAR_SET_SIZE {
            if self.has_child(i) {
                if let Some(child) = self.children[i].take() {
                    unsafe {
                        drop(Box::from_raw(child.as_ptr()));
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefix_matches() {
        let mut tree = RadixTree::<_, 64>::new();
        tree.insert("test", String::from("test_val"));
        tree.insert("testing", String::from("testing_val"));
        tree.insert("tent", String::from("tent_val"));

        // 测试最长前缀匹配
        assert_eq!(
            tree.longest_prefix_match("testing123"),
            Some((String::from("testing_val"), 7))
        );
        assert_eq!(
            tree.longest_prefix_match("test123"),
            Some((String::from("test_val"), 4))
        );

        // 测试第一个（最短）前缀匹配
        assert_eq!(
            tree.first_prefix_match("testing123"),
            Some((String::from("test_val"), 4))
        );

        // 测试所有前缀匹配
        assert_eq!(
            tree.all_prefix_matches("testing123"),
            vec![
                (String::from("test_val"), 4),
                (String::from("testing_val"), 7)
            ]
        );

        // 测试边界情况
        assert_eq!(tree.longest_prefix_match("te"), None);
        assert_eq!(tree.longest_prefix_match(""), None);
        assert_eq!(tree.longest_prefix_match("xyz"), None);

        // 测试精确匹配返回 None
        assert_eq!(tree.longest_prefix_match("test"), None);
        assert_eq!(tree.longest_prefix_match("testing"), None);

        // 测试没有重叠前缀的情况
        assert_eq!(
            tree.longest_prefix_match("tent123"),
            Some((String::from("tent_val"), 4))
        );
    }

    #[test]
    fn test_prefix_matches_with_shared_prefixes() {
        let mut tree = RadixTree::<_, 64>::new();
        tree.insert("a", String::from("a_val"));
        tree.insert("ab", String::from("ab_val"));
        tree.insert("abc", String::from("abc_val"));

        // 测试逐步增长的前缀匹配
        assert_eq!(
            tree.all_prefix_matches("abcd"),
            vec![
                (String::from("a_val"), 1),
                (String::from("ab_val"), 2),
                (String::from("abc_val"), 3)
            ]
        );

        // 验证最长前缀匹配返回最长的
        assert_eq!(
            tree.longest_prefix_match("abcd"),
            Some((String::from("abc_val"), 3))
        );
    }

    #[test]
    fn test_prefix_matches_with_non_string_values() {
        let mut tree = RadixTree::<_, 64>::new();
        tree.insert("one", 1);
        tree.insert("one_two", 12);
        tree.insert("one_two_three", 123);

        // 测试数字类型的前缀匹配
        assert_eq!(
            tree.all_prefix_matches("one_two_three_four"),
            vec![(1, 3), (12, 7), (123, 13)]
        );

        assert_eq!(
            tree.longest_prefix_match("one_two_three_four"),
            Some((123, 13))
        );
    }

    #[test]
    fn test_prefix_matches_with_removal() {
        let mut tree = RadixTree::<_,64>::new();

        // 插入测试数据
        tree.insert("test", String::from("test_val"));
        tree.insert("testing", String::from("testing_val"));
        tree.insert("tent", String::from("tent_val"));

        // 确认初始状态
        assert_eq!(
            tree.all_prefix_matches("testing123"),
            vec![
                (String::from("test_val"), 4),
                (String::from("testing_val"), 7)
            ]
        );

        // 删除中间节点
        tree.remove("test");

        // 验证删除后的前缀匹配
        assert_eq!(
            tree.all_prefix_matches("testing123"),
            vec![(String::from("testing_val"), 7)]
        );

        // 删除叶子节点
        tree.remove("testing");

        // 验证删除后没有匹配
        assert_eq!(tree.all_prefix_matches("testing123"), vec![]);
        assert_eq!(tree.longest_prefix_match("testing123"), None);

        // 验证其他分支不受影响
        assert_eq!(
            tree.longest_prefix_match("tent123"),
            Some((String::from("tent_val"), 4))
        );
    }

    #[test]
    fn test_prefix_matches_with_complex_removals() {
        let mut tree = RadixTree::<_, 64>::new();

        // 构建更复杂的前缀关系
        tree.insert("a", 1);
        tree.insert("ab", 2);
        tree.insert("abc", 3);
        tree.insert("abcd", 4);

        // 初始状态验证
        assert_eq!(
            tree.all_prefix_matches("abcde"),
            vec![(1, 1), (2, 2), (3, 3), (4, 4)]
        );

        // 删除中间节点
        tree.remove("ab");
        assert_eq!(
            tree.all_prefix_matches("abcde"),
            vec![(1, 1), (3, 3), (4, 4)]
        );

        // 删除根节点
        tree.remove("a");
        assert_eq!(
            tree.all_prefix_matches("abcde"),
            vec![(3, 3), (4, 4)]
        );

        // 删除所有剩余节点
        tree.remove("abc");
        tree.remove("abcd");
        assert_eq!(tree.all_prefix_matches("abcde"), vec![]);
    }

    #[test]
    fn test_prefix_matches_after_reinsert() {
        let mut tree = RadixTree::<_, 64>::new();

        // 初始插入
        tree.insert("test", 1);
        tree.insert("testing", 2);

        // 删除
        tree.remove("test");
        assert_eq!(
            tree.all_prefix_matches("testing123"),
            vec![(2, 7)]
        );

        // 重新插入
        tree.insert("test", 3);
        assert_eq!(
            tree.all_prefix_matches("testing123"),
            vec![(3, 4), (2, 7)]
        );
    }

    #[test]
    fn test_prefix_matches_with_value_replacement() {
        let mut tree = RadixTree::<_, 64>::new();

        // 初始插入
        tree.insert("test", String::from("original"));
        tree.insert("testing", String::from("testing_val"));

        // 替换值
        tree.insert("test", String::from("replaced"));

        // 验证前缀匹配使用了新值
        assert_eq!(
            tree.all_prefix_matches("testing123"),
            vec![
                (String::from("replaced"), 4),
                (String::from("testing_val"), 7)
            ]
        );
    }

    #[test]
    fn test_ascii_bitset_operations() {
        let mut tree = RadixTree::<_, 64>::new();

        // 1. 测试基础ASCII字符插入
        // 小写字母
        tree.insert("abc", 1);
        // 大写字母
        tree.insert("ABC", 2);
        // 数字
        tree.insert("123", 3);
        // 特殊字符
        tree.insert("_-.", 4);

        // 2. 测试查找
        assert_eq!(tree.exact_match("abc"), Some(1));
        assert_eq!(tree.exact_match("ABC"), Some(2));
        assert_eq!(tree.exact_match("123"), Some(3));
        assert_eq!(tree.exact_match("_-."), Some(4));

        // 3. 测试分裂节点场景
        // 插入共享前缀的字符串，触发节点分裂
        tree.insert("abcd", 5);
        tree.insert("ABCd", 6);

        assert_eq!(tree.exact_match("abc"), Some(1));
        assert_eq!(tree.exact_match("abcd"), Some(5));
        assert_eq!(tree.exact_match("ABC"), Some(2));
        assert_eq!(tree.exact_match("ABCd"), Some(6));

        // 4. 测试删除和合并
        // 删除叶子节点
        tree.remove("abcd");
        assert_eq!(tree.exact_match("abcd"), None);
        assert_eq!(tree.exact_match("abc"), Some(1));

        // 删除中间节点后重新插入
        tree.remove("abc");
        assert_eq!(tree.exact_match("abc"), None);
        tree.insert("abc", 7);
        assert_eq!(tree.exact_match("abc"), Some(7));

        // 5. 测试混合字符串操作
        let mixed = "Test-123_ABC.txt";
        tree.insert(mixed, 8);
        assert_eq!(tree.exact_match(mixed), Some(8));

        // 6. 测试前缀匹配
        assert_eq!(
            tree.all_prefix_matches("Test-123_ABC.txt.bak")
                .into_iter()
                .map(|(v, _)| v)
                .collect::<Vec<_>>(),
            vec![8]
        );

        // 7. 测试复杂的删除和重组
        tree.insert("Test", 9);
        tree.insert("Test-", 10);
        tree.insert("Test-123", 11);

        // 验证子节点计数
        unsafe {
            let node = tree.find_node("Test").unwrap();
            assert_eq!((*node.as_ptr()).count_children(), 1);
        }

        // 删除中间节点，测试合并
        tree.remove("Test-");

        // 验证结构完整性
        assert_eq!(tree.exact_match("Test"), Some(9));
        assert_eq!(tree.exact_match("Test-"), None);
        assert_eq!(tree.exact_match("Test-123"), Some(11));
        assert_eq!(tree.exact_match("Test-123_ABC.txt"), Some(8));

        // 8. 测试边界情况
        // 空字符串
        tree.insert("", 12);
        assert_eq!(tree.exact_match(""), Some(12));

        // 单字符
        tree.insert("a", 13);
        tree.insert("A", 14);
        tree.insert("1", 15);
        tree.insert(".", 16);

        assert_eq!(tree.exact_match("a"), Some(13));
        assert_eq!(tree.exact_match("A"), Some(14));
        assert_eq!(tree.exact_match("1"), Some(15));
        assert_eq!(tree.exact_match("."), Some(16));

        // 9. 验证树的大小
        assert!(tree.len() > 0);

        // 10. 清空树
        tree.clear();
        assert_eq!(tree.len(), 0);
        assert_eq!(tree.exact_match("abc"), None);
    }

    #[test]
    fn test_node_bitset_operations() {
        let mut node = Node::<i32, 64>::new();

        // 测试位操作
        let indices = vec![
            0,  // 'a'
            25, // 'z'
            26, // 'A'
            51, // 'Z'
            52, // '.'
            53, // '_'
            54, // '-'
            55  // '/'
        ];

        // 设置位
        for &idx in &indices {
            node.set_child(idx);
            assert!(node.has_child(idx));
        }

        // 验证子节点计数
        assert_eq!(node.count_children(), indices.len() as u32);

        // 清除位
        for &idx in &indices {
            node.clear_child(idx);
            assert!(!node.has_child(idx));
        }

        // 验证清除后的计数
        assert_eq!(node.count_children(), 0);

        // 测试单个子节点的情况
        node.set_child(0);
        assert_eq!(node.count_children(), 1);
        assert!(node.get_only_child().is_none()); // 因为children数组是空的

        // 验证can_merge_with_child逻辑
        assert!(node.value.is_none());
        assert_eq!(node.count_children(), 1);
        assert_eq!(node.edge_len, 0);
    }

    #[test]
    fn test_split_behavior() {
        let mut tree = RadixTree::<_, 64>::new();

        // 插入第一个字符串
        tree.insert("abc", 1);

        // 在分裂前检查状态
        unsafe {
            let node = tree.find_node("abc").unwrap();
            println!("Before split:");
            println!("Edge: {:?}", (*node.as_ptr()).edge_str());
            println!("Value: {:?}", (*node.as_ptr()).value);
            println!("Children mask: {:?}", (*node.as_ptr()).children_mask);
        }

        // 触发分裂
        tree.insert("abcd", 2);

        // 检查分裂后的状态
        unsafe {
            // 检查 "abc" 节点
            let abc_node = tree.find_node("abc").unwrap();
            println!("After split - abc node:");
            println!("Edge: {:?}", (*abc_node.as_ptr()).edge_str());
            println!("Value: {:?}", (*abc_node.as_ptr()).value);
            println!("Children mask: {:?}", (*abc_node.as_ptr()).children_mask);

            // 检查 "abcd" 节点
            let abcd_node = tree.find_node("abcd").unwrap();
            println!("After split - abcd node:");
            println!("Edge: {:?}", (*abcd_node.as_ptr()).edge_str());
            println!("Value: {:?}", (*abcd_node.as_ptr()).value);
            println!("Children mask: {:?}", (*abcd_node.as_ptr()).children_mask);
        }

        assert_eq!(tree.exact_match("abc"), Some(1));
        assert_eq!(tree.exact_match("abcd"), Some(2));
    }

    #[test]
    fn test_reference_prefix_matches() {
        let mut tree = RadixTree::<String, 64>::new();
        tree.insert("test", String::from("test_val"));
        tree.insert("testing", String::from("testing_val"));
        tree.insert("tent", String::from("tent_val"));

        // Test immutable reference matches
        if let Some((val_ref, len)) = tree.first_prefix_match_ref("testing123") {
            assert_eq!(val_ref, "test_val");
            assert_eq!(len, 4);
        } else {
            panic!("Expected to find a match");
        }

        let all_refs = tree.all_prefix_matches_ref("testing123");
        assert_eq!(all_refs.len(), 2);
        assert_eq!(all_refs[0].0, "test_val");
        assert_eq!(all_refs[0].1, 4);
        assert_eq!(all_refs[1].0, "testing_val");
        assert_eq!(all_refs[1].1, 7);

        if let Some((val_ref, len)) = tree.longest_prefix_match_ref("testing123") {
            assert_eq!(val_ref, "testing_val");
            assert_eq!(len, 7);
        } else {
            panic!("Expected to find a match");
        }

        // Test exact match reference
        assert_eq!(tree.exact_match_ref("test").unwrap(), "test_val");
        assert_eq!(tree.exact_match_ref("missing"), None);

        // Test mutable reference matches
        if let Some((val_ref, len)) = tree.first_prefix_match_mut("testing123") {
            assert_eq!(*val_ref, "test_val");
            *val_ref = String::from("modified_test_val");
            assert_eq!(len, 4);
        }

        // Check if modification was successful
        assert_eq!(tree.exact_match_ref("test").unwrap(), "modified_test_val");

        // Test all mutable references
        {
            let mut all_mut_refs = tree.all_prefix_matches_mut("testing123");
            assert_eq!(all_mut_refs.len(), 2);

            // Modify values via mutable references
            *all_mut_refs[0].0 = String::from("modified_again_test_val");
            *all_mut_refs[1].0 = String::from("modified_testing_val");
        }

        // Verify modifications
        assert_eq!(tree.exact_match_ref("test").unwrap(), "modified_again_test_val");
        assert_eq!(tree.exact_match_ref("testing").unwrap(), "modified_testing_val");

        // Test longest mutable reference
        if let Some((val_ref, _)) = tree.longest_prefix_match_mut("testing123") {
            *val_ref = String::from("final_testing_val");
        }

        // Verify final modification
        assert_eq!(tree.exact_match_ref("testing").unwrap(), "final_testing_val");
    }

    #[test]
    fn test_reference_matches_with_types_without_clone() {
        // Define a type that doesn't implement Clone
        struct NoClone {
            value: i32,
        }

        let mut tree = RadixTree::<NoClone, 64>::new();
        tree.insert("one", NoClone { value: 1 });
        tree.insert("two", NoClone { value: 2 });
        tree.insert("three", NoClone { value: 3 });

        // Test immutable reference methods
        if let Some((val_ref, _)) = tree.first_prefix_match_ref("one_more") {
            assert_eq!(val_ref.value, 1);
        } else {
            panic!("Expected to find a match");
        }

        let refs = tree.all_prefix_matches_ref("one_more");
        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0].0.value, 1);

        // Test mutable reference methods
        if let Some((val_ref, _)) = tree.first_prefix_match_mut("two_more") {
            val_ref.value = 22;
        }

        // Verify modification
        assert_eq!(tree.exact_match_ref("two").unwrap().value, 22);

        // Test with multiple prefix matches
        let mut nested_tree = RadixTree::<NoClone, 64>::new();
        nested_tree.insert("a", NoClone { value: 1 });
        nested_tree.insert("ab", NoClone { value: 2 });
        nested_tree.insert("abc", NoClone { value: 3 });
        nested_tree.insert("abce", NoClone { value: 4 });

        let all_refs = nested_tree.all_prefix_matches_ref("abcd");
        assert_eq!(all_refs.len(), 3);
        assert_eq!(all_refs[0].0.value, 1);
        assert_eq!(all_refs[1].0.value, 2);
        assert_eq!(all_refs[2].0.value, 3);

        // Modify all values
        {
            let mut all_mut_refs = nested_tree.all_prefix_matches_mut("abcd");
            for (i, (val_ref, _)) in all_mut_refs.iter_mut().enumerate() {
                val_ref.value = (i as i32 + 1) * 10;
            }
        }

        // Verify modifications
        assert_eq!(nested_tree.exact_match_ref("a").unwrap().value, 10);
        assert_eq!(nested_tree.exact_match_ref("ab").unwrap().value, 20);
        assert_eq!(nested_tree.exact_match_ref("abc").unwrap().value, 30);
    }


    #[test]
    fn test_entries() {
        let mut tree = RadixTree::<String, 64>::new();
        tree.insert("test", String::from("test_val"));
        tree.insert("testing", String::from("testing_val"));
        tree.insert("tent", String::from("tent_val"));

        let entries = tree.entries();
        assert_eq!(entries.len(), 3);
        assert!(entries.contains(&("test".to_string(), &"test_val".to_string())));
        assert!(entries.contains(&("testing".to_string(), &"testing_val".to_string())));
        assert!(entries.contains(&("tent".to_string(), &"tent_val".to_string())));
    }

    #[test]
    fn test_clear_with_dump() {
        let mut tree = RadixTree::<String, 64>::new();
        tree.insert("test", String::from("test_val"));
        tree.insert("testing", String::from("testing_val"));
        tree.insert("tent", String::from("tent_val"));

        let dumped_entries = tree.clear_and_dump();
        assert_eq!(dumped_entries.len(), 3);
        assert!(dumped_entries.contains(&("test".to_string(), "test_val".to_string())));
        assert!(dumped_entries.contains(&("testing".to_string(), "testing_val".to_string())));
        assert!(dumped_entries.contains(&("tent".to_string(), "tent_val".to_string())));

        assert!(tree.is_empty());
    }

}