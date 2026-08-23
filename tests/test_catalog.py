import gzip
import json

import pandas as pd
import pytest

from stylist.catalog import (
    build_doc_text,
    derive_audience,
    group_key,
    ingest,
    normalize_record,
    parse_price,
    select_rows,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        (12.5, (12.5, "float")),
        (7, (7.0, "float")),
        ("$12.99", (12.99, "string")),
        ("12.99", (12.99, "string")),
        ("$10 - $20", (None, "range")),
        (None, (None, "none")),
        ("", (None, "none")),
        ("abc", (None, "unparsed")),
        (-3.0, (None, "unparsed")),
        (float("nan"), (None, "none")),
    ],
)
def test_parse_price(raw, expected):
    assert parse_price(raw) == expected


@pytest.mark.parametrize(
    "title,department,expected",
    [
        ("Crew Socks for Men", "womens", "women"),  # department wins over title
        ("Anything", "mens", "men"),
        ("Anything", "unisex-adult", "unisex"),
        ("Anything", "girls", "girls"),
        ("Anything", "baby-boys", "baby"),
        ("Men's Slim Fit Chino", None, "men"),
        ("Womens Palazzo Pants", None, "women"),
        ("Ladies Summer Dress", None, "women"),
        ("Girls' Trapeze Dress", None, "girls"),
        ("Boys Swim Trunks", None, "boys"),
        ("Toddler Baby Boy Shirt", None, "baby"),
        ("Unisex Ear Warmers", None, "unisex"),
        ("Beanie for Men Women", None, "unisex"),
        ("Plain Beanie", None, "unknown"),
    ],
)
def test_derive_audience(title, department, expected):
    assert derive_audience(title, department) == expected


def test_group_key_strips_variant_suffixes():
    a = "Ecetana Sandals for Women Casual Summer (Black, 8)"
    b = "Ecetana Sandals for Women Casual Summer (Blue, 10)"
    assert group_key(a) == group_key(b) == "ecetana sandals for women casual summer"


def test_group_key_strips_trailing_size_words():
    a = "iloveSIA Mens Hiking Walking Leather Sandals Brown US Size 8"
    b = "iloveSIA Mens Hiking Walking Leather Sandals Brown US Size 10"
    assert group_key(a) == group_key(b)


def test_group_key_keeps_different_products_apart():
    assert group_key("Nike Air Zoom Pegasus 38") != group_key("Nike Air Zoom Pegasus 39")


def test_build_doc_text_uses_title_features_and_useful_details_only():
    rec = {
        "title": "Cozy Sweater",
        "features": ["Machine Wash", "Pull On closure", "x", "y", "z"],
        "details": {
            "Material": "Cotton",
            "Package Dimensions": "10 x 10 x 1 inches",
            "Department": "womens",
        },
        "store": "Acme",
        "description": [],
    }
    text = build_doc_text(normalize_record(rec, 0))
    assert text.startswith("Cozy Sweater")
    assert "Machine Wash" in text and "Cotton" in text and "Acme" in text
    assert "Package Dimensions" not in text and "10 x 10" not in text
    assert "z" not in text.split("|")[1]  # only first 4 features kept
    assert len(text) <= 600


def test_build_doc_text_is_capped_at_600_chars():
    rec = {"title": "t" * 50, "features": ["f" * 200] * 4, "details": {}, "store": "s" * 300}
    assert len(build_doc_text(normalize_record(rec, 0))) <= 600


def test_normalize_record_flattens_and_derives_fields():
    raw = {
        "main_category": "AMAZON FASHION",
        "title": "DouBCQ Women's Palazzo Pants (Blue, XL)",
        "average_rating": 4.1,
        "rating_number": 7,
        "features": ["Drawstring closure"],
        "description": ["Flowy and light"],
        "price": None,
        "images": [{"thumb": "t.jpg", "large": "l.jpg", "hi_res": None, "variant": "MAIN"}],
        "videos": [],
        "store": "DouBCQ",
        "categories": [],
        "details": {"Department": "womens", "Material": "Polyester"},
        "parent_asin": "B08R39MRDW",
        "bought_together": None,
    }
    row = normalize_record(raw, row_id=17)
    assert row["row_id"] == 17
    assert row["parent_asin"] == "B08R39MRDW"
    assert row["price"] is None and row["price_status"] == "none"
    assert row["audience"] == "women"
    assert row["material"] == "Polyester"
    assert row["image_url"] == "l.jpg"
    assert "Drawstring closure" in row["doc_text"]
    assert row["description"] == "Flowy and light"


def _write_raw(path, records):
    with gzip.open(path, "wt") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _raw(i, title, rating_number=1, price=None, dept=None):
    return {
        "main_category": "AMAZON FASHION",
        "title": title,
        "average_rating": 4.0,
        "rating_number": rating_number,
        "features": [],
        "description": [],
        "price": price,
        "images": [{"thumb": f"t{i}.jpg", "large": f"l{i}.jpg", "hi_res": None, "variant": "MAIN"}],
        "videos": [],
        "store": "S",
        "categories": [],
        "details": {"Department": dept} if dept else {},
        "parent_asin": f"ASIN{i}",
        "bought_together": None,
    }


def test_ingest_writes_parquet_with_stats(tmp_path):
    raw = tmp_path / "raw.jsonl.gz"
    _write_raw(raw, [_raw(0, "Men's Boots", 5, 20.0, "mens"), _raw(1, "Women Dress", 50)])
    out = tmp_path / "catalog.parquet"
    stats = ingest(raw, out)
    df = pd.read_parquet(out)
    assert list(df["row_id"]) == [0, 1]
    assert stats.rows == 2
    assert stats.price_status["float"] == 1 and stats.price_status["none"] == 1
    assert stats.coverage["image_url"] == 1.0
    assert "by_rating_bucket" in stats.as_dict()


def test_ingest_respects_limit(tmp_path):
    raw = tmp_path / "raw.jsonl.gz"
    _write_raw(raw, [_raw(i, f"Item {i}") for i in range(5)])
    out = tmp_path / "catalog.parquet"
    ingest(raw, out, limit=3)
    assert len(pd.read_parquet(out)) == 3


def test_select_rows_popular_takes_most_rated_in_row_order():
    df = pd.DataFrame({"row_id": [0, 1, 2, 3], "rating_number": [1, 50, 10, 99]})
    out = select_rows(df, limit=2, sampling="popular")
    assert list(out["row_id"]) == [1, 3]


def test_select_rows_random_is_seeded_and_sorted():
    df = pd.DataFrame({"row_id": list(range(100)), "rating_number": [1] * 100})
    a = select_rows(df, limit=10, sampling="random", seed=42)
    b = select_rows(df, limit=10, sampling="random", seed=42)
    assert list(a["row_id"]) == list(b["row_id"])
    assert list(a["row_id"]) == sorted(a["row_id"])


def test_select_rows_all_ignores_limit():
    df = pd.DataFrame({"row_id": [0, 1, 2], "rating_number": [1, 2, 3]})
    assert len(select_rows(df, limit=1, sampling="all")) == 3


def test_select_rows_rejects_unknown_sampling():
    df = pd.DataFrame({"row_id": [0], "rating_number": [1]})
    with pytest.raises(ValueError):
        select_rows(df, limit=1, sampling="best")


def test_group_key_strips_trailing_comma_variant_segments():
    a = "Women Floral Printe Swimsuit Summer Beach Bathing Suits Push Up Brazilian Suit, Multi/Flor"
    b = "Women Floral Printe Swimsuit Summer Beach Bathing Suits Push Up Brazilian Suit, M/US 4-6"
    c = "JOSIFER Women Summer Swimsuit Coverups Crochet Cover up, Black, Large"
    d = "JOSIFER Women Summer Swimsuit Coverups Crochet Cover up, White, Small"
    assert group_key(a) == group_key(b)
    assert group_key(c) == group_key(d)
    assert group_key(c).endswith("cover up")


def test_group_key_keeps_descriptive_comma_segments():
    t = "Clear Crossbody Purse Bag, Clear Stadium Bag with Adjustable Shoulder Strap"
    assert "stadium bag" in group_key(t)


def test_load_catalog_subset_matches_select_rows(fixture_catalog, fixture_catalog_df):
    from stylist.catalog import load_catalog_subset

    sub = load_catalog_subset(fixture_catalog, limit=25, sampling="popular")
    expected = select_rows(fixture_catalog_df, limit=25, sampling="popular")
    assert list(sub["row_id"]) == list(expected["row_id"])
    assert list(sub.columns) == list(fixture_catalog_df.columns)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1,299", (1299.0, "string")),
        ("$1,299.50", (1299.5, "string")),
        ("12,99", (1299.0, "string")),  # comma = thousands separator (US listings)
        (float("inf"), (None, "unparsed")),
    ],
)
def test_parse_price_thousands_and_non_finite(raw, expected):
    assert parse_price(raw) == expected


def test_normalize_record_guards_non_finite_rating():
    raw = {"title": "x", "average_rating": float("nan"), "rating_number": 3, "parent_asin": "A"}
    row = normalize_record(raw, row_id=1)
    assert row["average_rating"] is None


def test_group_key_strips_nested_parenthetical_suffix():
    a = "Wanlorraiy Women's Rhinestone Flat Sandals Ankle Strap Flat Shoes(8 B(M) US,Silver)"
    b = "Wanlorraiy Women's Rhinestone Flat Sandals Ankle Strap Flat Shoes(10 B(M) US,Black)"
    assert group_key(a) == group_key(b)
    assert group_key(a).endswith("flat shoes")


def test_group_key_strips_trailing_shoe_sizes_but_keeps_model_numbers():
    a = "OUOUVALLEY Lace Up Patent Leather Oxford Dress Shoes Formal Wedding Shoes 8"
    b = "OUOUVALLEY Lace Up Patent Leather Oxford Dress Shoes Formal Wedding Shoes 10.5"
    assert group_key(a) == group_key(b) == group_key(a[:-2])
    assert group_key("Nike Air Zoom Pegasus 38") != group_key("Nike Air Zoom Pegasus 39")
    assert group_key("Levi's 501 Original Fit Jeans").endswith("jeans")


@pytest.mark.parametrize(
    "raw,expected",
    [(5, 5), ("12", 12), (-3, 0), (float("inf"), 0), ("x", 0), (None, 0), (2**40, 2**31 - 1)],
)
def test_rating_number_is_a_bounded_int(raw, expected):
    row = normalize_record({"title": "t", "rating_number": raw, "parent_asin": "A"}, row_id=1)
    assert row["rating_number"] == expected
