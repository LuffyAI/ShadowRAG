CREATE TABLE IF NOT EXISTS entities (
        id TEXT NOT NULL,
        ename TEXT NOT NULL,
        etype TEXT NOT NULL,
        edesc TEXT NOT NULL,
        PRIMARY KEY (id)
    );

CREATE TABLE IF NOT EXISTS shadow_entity(
    shadow_id TEXT NOT NULL,
    PRIMARY KEY (shadow_id)
);

CREATE TABLE IF NOT EXISTS shadow_entity_to_entity(
    shadow_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    FOREIGN KEY (shadow_id) REFERENCES shadow_entity(shadow_id),
    FOREIGN KEY (entity_id) REFERENCES entities(id)
);

CREATE TABLE IF NOT EXISTS text_chunks (
    chunk_id TEXT NOT NULL,
    content TEXT NOT NULL,
    PRIMARY KEY (chunk_id)
);

CREATE TABLE IF NOT EXISTS marked_chunks (
    chunk_id TEXT NOT NULL,
    content TEXT NOT NULL,
    FOREIGN KEY (chunk_id) REFERENCES text_chunks(chunk_id)
);

CREATE TABLE IF NOT EXISTS shadow_entity_chunks (
    chunk_id TEXT NOT NULL,
    content TEXT NOT NULL,
    FOREIGN KEY (chunk_id) REFERENCES text_chunks(chunk_id)
);


CREATE TABLE IF NOT EXISTS shadow_chunks(
    chunk_id TEXT NOT NULL UNIQUE,
    shadow_content TEXT NOT NULL,
    FOREIGN KEY (chunk_id) REFERENCES text_chunks(chunk_id)
);


CREATE TABLE IF NOT EXISTS entity_chunk_pairs(
    entity_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    PRIMARY KEY (entity_id, chunk_id),
    FOREIGN KEY (entity_id) REFERENCES entities(id),
    FOREIGN KEY (chunk_id) REFERENCES text_chunks(chunk_id)
);

CREATE TABLE IF NOT EXISTS action_chunk_pairs(
    action_span TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    shadow_action TEXT NOT NULL,
    PRIMARY KEY (shadow_action, chunk_id),
    FOREIGN KEY (chunk_id) REFERENCES text_chunks(chunk_id)
);


CREATE TABLE IF NOT EXISTS query_cache (
    query_hash TEXT NOT NULL PRIMARY KEY,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    embedding BLOB
);

