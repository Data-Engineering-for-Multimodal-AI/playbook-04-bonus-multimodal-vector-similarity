# Sample data

The book's author, Vasundra Srinivasan, created the videos and images here with a Google generative video model, and generated the order records. They show no real restaurant, kitchen, robot, customer or person. Like the code, they are released under the MIT License (see [LICENSE](../LICENSE)).

| Files | What they are | Used by |
|---|---|---|
| `videos/download.mp4`, `videos/download (1).mp4` … `videos/download (31).mp4` (32 files, 54 MB) | Food-preparation clips. MP4 (H.264), 1280×720, 24 fps, 5.0 s each | Embedding generator, similarity search, quality checks, dashboard |
| `images/Crispy_Shoyu.jpg`, `House_Salmon.jpg`, `Kid's_Crab_Bowl.jpg`, `Miso_Chicken.jpg`, `Miso_Salmon.jpg`, `Salmon_Edamame.jpg`, `Tuna_Lover.jpg` (7 files) | Reference photos of finished bowls, 2500×1667 | Embedding generator, quality checks (the reference images), dashboard |
| `orders/multirestaurant_orders.jsonl` (500 records) | Invented delivery orders for three fictional restaurants (Sample Seafood House, Sample Poke Bar, Sample Bowl Kitchen): items, prices, discounts, timings, ratings and reviews | Not read by the code; an example of the order data that sits next to the media in this pipeline |
| `sample_queries.json` | Example search queries, written by `python src/utils/init.py --generate-queries` | `SimilaritySearch.batch_search` (optional) |

The screenshots and frames in [`docs/images/`](../docs/images) come from the author's runs of the dashboard on these files and are MIT-licensed as well.

## Generated folders (not in git)

The pipeline writes these; delete them to start over.

- `embeddings/`: one `<name>_embedding.npy` (512 numbers) and `<name>_metadata.json` per video and image, plus a `batch_summary_<time>.json` per run.
- `quality/`: one `<video>_quality.json` report per video checked (a later check of the same video replaces it).
