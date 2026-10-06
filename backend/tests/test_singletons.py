"""The clients are created once per process, not per request."""


def test_chroma_handle_is_cached():
    from app.database.chroma import get_database

    assert get_database() is get_database()


def test_generator_is_cached():
    from app.rag.generator import get_generator

    assert get_generator() is get_generator()


def test_embedding_model_is_cached():
    from app.rag.embeddings import get_embedding_model

    assert get_embedding_model() is get_embedding_model()


def test_pipeline_is_cached():
    from app.rag.pipeline import get_pipeline

    assert get_pipeline() is get_pipeline()


def test_pipeline_reuses_the_shared_database():
    from app.database.chroma import get_database
    from app.rag.pipeline import get_pipeline

    assert get_pipeline().retriever.database is get_database()


def test_cache_clear_rebuilds_the_handle():
    from app.database.chroma import get_database

    first = get_database()

    get_database.cache_clear()

    assert get_database() is not first
