import click
import openai
from pybtex.database.input import bibtex
import pandas as pd
import numpy as np
from pathlib import Path


def read_bib(file_name):
    entries = bibtex.Parser().parse_file(file_name).entries.values()
    data = [{**entry.fields, 'type': entry.type} for entry in entries]
    return pd.DataFrame(data)

def embed_text(texts):
    return openai.Embedding.create(
        input=texts,
        model="text-embedding-ada-002",
        )["data"]


@click.command()
@click.argument('bibfile')
@click.argument('out_dir')
def main(bibfile, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    df = read_bib(bibfile)

    titles = df['title'].tolist()
    title_embeddings = embed_text(titles)

    abstracts = df['abstract'].tolist()
    abstract_embeddings = embed_text(abstracts)

    title_emb = np.array([openai_obj['embedding'] for openai_obj in title_embeddings])
    np.save(out_dir / 'title_emb.npy', title_emb)

    abst_emb = np.array([openai_obj['embedding'] for openai_obj in abstract_embeddings])
    np.save(out_dir / 'abst_emb.npy', abst_emb)

    df.to_feather(out_dir / 'bibliography.feather')


if __name__ == '__main__':
    main()
    # from IPython import embed
    # main('acm.bib', 'CHI23')

    # embed()
