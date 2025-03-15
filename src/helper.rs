// 扩展字符集大小到128以支持更多URL合法字符
pub const CHAR_SET_SIZE: usize = 128;

// According to rfc of dns, the max length of domain is 256, the max length of each label of domain is 63
pub const DOMAIN_PATH_LEN: usize = 64;
pub const IPV4_PATH_LEN: usize = 32;
pub const IPV6_PATH_LEN: usize = 128;

#[inline]
pub fn char_to_index(c: char) -> Option<usize> {
    match c {
        'a'..='z' => Some((c as usize) - ('a' as usize)),
        'A'..='Z' => Some(26 + (c as usize) - ('A' as usize)),
        '.' => Some(52),
        '_' => Some(53),
        '-' => Some(54),
        '/' => Some(55),

        '0'..='9' => Some(56 + (c as usize) - ('0' as usize)), // 56-65
        ':' => Some(66),
        '?' => Some(67),
        '#' => Some(68),
        '[' => Some(69),
        ']' => Some(70),
        '@' => Some(71),
        '!' => Some(72),
        '$' => Some(73),
        '&' => Some(74),
        '\'' => Some(75),
        '(' => Some(76),
        ')' => Some(77),
        '*' => Some(78),
        '+' => Some(79),
        ',' => Some(80),
        ';' => Some(81),
        '=' => Some(82),
        '%' => Some(83),
        '~' => Some(84),
        _ => None,
    }
}