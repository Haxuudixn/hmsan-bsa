from bid_slicing.data.label_schema import LabelSchema


def test_label_schema_encodes_known_and_unknown_labels():
    schema = LabelSchema(
        class_labels=["封面页", "目录"],
        boundary_labels=["O", "B-SECTION"],
        ignore_index=-100,
        mixed_section_label="MIXED",
    )

    assert schema.encode_class("封面页") == 0
    assert schema.encode_class("目录") == 1
    assert schema.encode_class("") == -100
    assert schema.encode_class(None) == -100
    assert schema.encode_class("不存在") == -100


def test_boundary_and_section_encoding():
    schema = LabelSchema(
        class_labels=["封面页", "目录"],
        boundary_labels=["O", "B-SECTION", "I-SECTION", "E-SECTION"],
        ignore_index=-100,
        mixed_section_label="MIXED",
    )

    assert schema.encode_boundary("O") == 0
    assert schema.encode_boundary("B-SECTION") == 1
    assert schema.encode_boundary("BAD") == 0
    assert schema.num_classes == 2
    assert schema.num_boundaries == 4
