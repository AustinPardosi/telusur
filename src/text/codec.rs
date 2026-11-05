//! Varints and Lucene's SmallFloat doc length norms.

/// LEB128. A value of 1 or more never encodes to a 0x00 byte.
pub fn put_varint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push(v as u8 | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

/// Reads the varint at `*pos` and moves `*pos` past it.
pub fn get_varint(buf: &[u8], pos: &mut usize) -> u64 {
    let mut v = 0;
    let mut shift = 0;
    loop {
        let b = buf[*pos];
        *pos += 1;
        v |= u64::from(b & 0x7f) << shift;
        if b < 0x80 {
            return v;
        }
        shift += 7;
    }
}

/// Values below this are stored as is: `255 - longToInt4(i32::MAX)` in Lucene.
const FREE: u32 = 24;

/// Lucene's `SmallFloat.intToByte4`. Exact up to 40, then keeps 4 significant bits of
/// `len - 24`, rounding down.
pub fn int_to_byte4(len: u32) -> u8 {
    let i = len.min(i32::MAX as u32);
    if i < FREE {
        return i as u8;
    }
    let i = i - FREE;
    let bits = 32 - i.leading_zeros();
    let int4 = if bits < 4 {
        i
    } else {
        let shift = bits - 4;
        ((i >> shift) & 0x07) | ((shift + 1) << 3)
    };
    (FREE + int4) as u8
}

/// Lucene's `SmallFloat.byte4ToInt`.
pub fn byte4_to_int(b: u8) -> u32 {
    let i = u32::from(b);
    if i < FREE {
        return i;
    }
    let i = i - FREE;
    let bits = i & 0x07;
    FREE + match i >> 3 {
        0 => bits,
        e => (bits | 0x08) << (e - 1),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varint_round_trip_and_no_zero_bytes() {
        let values = [1, 127, 128, 300, 16_383, 16_384, u32::MAX as u64, u64::MAX];
        let mut buf = Vec::new();
        for v in values {
            put_varint(&mut buf, v);
        }
        assert!(!buf.contains(&0));
        let mut pos = 0;
        for v in values {
            assert_eq!(get_varint(&buf, &mut pos), v);
        }
        assert_eq!(pos, buf.len());
    }

    #[test]
    fn small_float_matches_lucene() {
        // Worked by hand from Lucene's SmallFloat.intToByte4 / byte4ToInt.
        for (len, decoded) in [
            (0, 0),
            (23, 23),
            (24, 24),
            (40, 40),
            (41, 40),
            (42, 42),
            (100, 96),
        ] {
            assert_eq!(byte4_to_int(int_to_byte4(len)), decoded, "len {len}");
        }
        assert_eq!(int_to_byte4(i32::MAX as u32), 255);
        assert_eq!(int_to_byte4(u32::MAX), 255);
        for b in 0..=255u8 {
            assert_eq!(int_to_byte4(byte4_to_int(b)), b);
        }
    }
}
