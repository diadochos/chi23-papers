## Dependencies
```
$ poetry self install
$ poetry install
```

## Manual
1. Get `.bib` from a conference's proceeding.
1. Convert it to a set of embeddings etc. by `convert.py`:
   `$ poetry run python convert.py <bib> <output_directory>`
1. Run streamlit: `$ poetry run streamlit run app.py`.

## Explanation
Environment variables for the `OPENAI_API_KEY`, etc., are set via `poetry-dotenv-plugin`.
Thus, they are written in `.env` files and are automatically loaded when running `poetry run`.

## acm.bib
https://dl.acm.org/doi/proceedings/10.1145/3544548

## Credit
https://github.com/kotarotanahashi/cvpr
