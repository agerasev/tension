use std::{
    fmt::Debug,
    ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive},
};

pub trait Index {
    fn saturating_wrap(&self, len: usize) -> usize;
}

impl Index for usize {
    fn saturating_wrap(&self, len: usize) -> usize {
        usize::min(*self, len)
    }
}

/// Index wrapper to allow indexaing from both ends.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum IndexFrom {
    /// Index from start.
    ///
    /// Just the standard indexing.
    Start(usize),

    /// Index from end.
    ///
    /// + `IndexFrom::End(len)` - first item.
    /// + `IndexFrom::End(1)` - last item.
    /// + `IndexFrom::End(0)` - item after last. Helpful for slicing, but accessing this item is OOB.
    End(usize),
}

impl Default for IndexFrom {
    fn default() -> Self {
        Self::Start(0)
    }
}

impl From<usize> for IndexFrom {
    fn from(value: usize) -> Self {
        Self::Start(value)
    }
}

impl Index for IndexFrom {
    fn saturating_wrap(&self, len: usize) -> usize {
        match self {
            Self::Start(i) => usize::min(*i, len),
            Self::End(i) => len.saturating_sub(*i),
        }
    }
}

pub trait IndexRange {
    fn canonicalize(&self) -> Range<IndexFrom>;
}

impl IndexRange for Range<usize> {
    fn canonicalize(&self) -> Range<IndexFrom> {
        IndexFrom::Start(self.start)..IndexFrom::Start(self.end)
    }
}
impl IndexRange for RangeInclusive<usize> {
    fn canonicalize(&self) -> Range<IndexFrom> {
        IndexFrom::Start(*self.start())..IndexFrom::Start(self.end() + 1)
    }
}
impl IndexRange for RangeFrom<usize> {
    fn canonicalize(&self) -> Range<IndexFrom> {
        IndexFrom::Start(self.start)..IndexFrom::End(0)
    }
}
impl IndexRange for RangeTo<usize> {
    fn canonicalize(&self) -> Range<IndexFrom> {
        IndexFrom::Start(0)..IndexFrom::Start(self.end)
    }
}
impl IndexRange for RangeToInclusive<usize> {
    fn canonicalize(&self) -> Range<IndexFrom> {
        IndexFrom::Start(0)..IndexFrom::Start(self.end + 1)
    }
}

impl IndexRange for Range<IndexFrom> {
    fn canonicalize(&self) -> Range<IndexFrom> {
        self.clone()
    }
}
impl IndexRange for RangeInclusive<IndexFrom> {
    fn canonicalize(&self) -> Range<IndexFrom> {
        *self.start()..match self.end() {
            IndexFrom::Start(i) => IndexFrom::Start(i + 1),
            IndexFrom::End(i) => IndexFrom::End(i - 1),
        }
    }
}
impl IndexRange for RangeFrom<IndexFrom> {
    fn canonicalize(&self) -> Range<IndexFrom> {
        self.start..IndexFrom::End(0)
    }
}
impl IndexRange for RangeTo<IndexFrom> {
    fn canonicalize(&self) -> Range<IndexFrom> {
        IndexFrom::Start(0)..self.end
    }
}
impl IndexRange for RangeToInclusive<IndexFrom> {
    fn canonicalize(&self) -> Range<IndexFrom> {
        IndexFrom::Start(0)..match self.end {
            IndexFrom::Start(i) => IndexFrom::Start(i + 1),
            IndexFrom::End(i) => IndexFrom::End(i - 1),
        }
    }
}

impl IndexRange for RangeFull {
    fn canonicalize(&self) -> Range<IndexFrom> {
        IndexFrom::Start(0)..IndexFrom::End(0)
    }
}

/// Marker for adding a new dimension.
#[derive(Clone, Copy, Default, Debug)]
pub struct NewAxis;
