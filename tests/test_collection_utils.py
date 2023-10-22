from torchbricks.collection_utils import unflatten


def test_unflatten_no_expansion():
    result = unflatten({'a': 12, 'b': 13, 'c': 14}, sep='.')
    expected = {'a': 12, 'b': 13, 'c': 14}
    assert result == expected

def test_unflatten_dotted_path_expansion():
    result = unflatten({'a.b.c': 12}, sep='.')
    expected = {'a': {'b': {'c': 12}}}
    assert result == expected

def test_unflatten_merging():
    result = unflatten({'a.b.c': 12, 'a': {'b.d': 13}}, sep='.')
    expected = {'a': {'b': {'c': 12, 'd': 13}}}
    assert result == expected

def test_unflatten_insertion_order_overwrites():
    result = unflatten({'a.b': 12, 'a': {'b': 13}}, sep='.')
    expected = {'a': {'b': 13}}
    assert result == expected

def test_unflatten_empty_insertion():
    result = unflatten({'a': {}}, sep='.')
    expected = {'a': {}}
    assert result == expected
