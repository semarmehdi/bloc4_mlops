# On importe la fonction qu'on veut tester
from utils.transform import item_to_row


def test_une_ligne_simple():
    # ARRANGE : on prépare une entrée comme celle que renvoie l'API
    item = {"data": {"columns": ["Age", "Gender"], "data": [[36, "Female"]]}}
    rows = []

    # ACT : on exécute la fonction
    item_to_row(item, rows)

    # ASSERT : on vérifie le résultat
    assert rows == [{"Age": 36, "Gender": "Female"}]


def test_la_colonne_current_time_est_supprimee():
    # La fonction doit retirer "current_time" du résultat
    item = {"data": {"columns": ["Age", "current_time"], "data": [[36, "2026-01-01"]]}}
    rows = []

    item_to_row(item, rows)

    assert rows == [{"Age": 36}]
    assert "current_time" not in rows[0]


def test_payload_vide_n_ajoute_rien():
    # Si pas de colonnes/données, la fonction ne doit rien ajouter
    item = {"data": {"columns": [], "data": []}}
    rows = []

    item_to_row(item, rows)

    assert rows == []


def test_plusieurs_appels_s_accumulent():
    # rows est partagé : deux appels => deux lignes
    rows = []

    item_to_row({"data": {"columns": ["Age"], "data": [[36]]}}, rows)
    item_to_row({"data": {"columns": ["Age"], "data": [[41]]}}, rows)

    assert len(rows) == 2
    assert rows[0] == {"Age": 36}
    assert rows[1] == {"Age": 41}
