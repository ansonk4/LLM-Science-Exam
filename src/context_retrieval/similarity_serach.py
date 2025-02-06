# Purpose: This script is used to generate the context for the RAG model. It uses the top 5 most similar paragraphs from the CirrusWiki dataset for each prompt in the test set. The script uses the SentenceTransformer model to encode the prompt and the CirrusWiki paragraphs. The cosine similarity is calculated between the prompt and the CirrusWiki paragraphs. The top 5 most similar paragraphs are then selected and saved in a CSV file.

import pandas as pd
from glob import glob
import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from pathlib import Path
from joblib import Parallel, delayed
from tqdm.auto import tqdm

def cos_similarity_matrix(a: torch.Tensor, b: torch.Tensor):
    """Calculates cosine similarities between tensor a and b."""
    sim_mt = torch.mm(a, b.transpose(0, 1))
    return sim_mt

def get_topk(embeddings_from, embeddings_to, topk=1000, bs=512):
    chunk = bs
    embeddings_chunks = embeddings_from.split(chunk)
    vals = []
    inds = []
    for idx in range(len(embeddings_chunks)):
        cos_sim_chunk = cos_similarity_matrix(
            embeddings_chunks[idx].to(embeddings_to.device).half(), embeddings_to
        ).float()

        cos_sim_chunk = torch.nan_to_num(cos_sim_chunk, nan=0.0)

        topk = min(topk, cos_sim_chunk.size(1))
        vals_chunk, inds_chunk = torch.topk(cos_sim_chunk, k=topk, dim=1)
        vals.append(vals_chunk[:, :].detach().cpu())
        inds.append(inds_chunk[:, :].detach().cpu())

        del vals_chunk
        del inds_chunk
        del cos_sim_chunk

    vals = torch.cat(vals).detach().cpu()
    inds = torch.cat(inds).detach().cpu()

    return inds, vals


def insert_value_at(tensor, value, position):
    # Ensure the position is valid
    if position < 0 or position >= len(tensor):
        raise ValueError("Position should be between 0 and tensor length - 1.")

    # Slice the tensor into two parts
    left = tensor[:position]
    right = tensor[position:]

    # Create a tensor for the value to be inserted
    value_tensor = torch.tensor([value], dtype=tensor.dtype)

    # Concatenate the tensors together and slice to the original length
    result = torch.cat([left, value_tensor, right])[:-1]

    return result


def insert_value_at_list(lst, value, position):
    # Ensure the position is valid
    if position < 0 or position >= len(lst):
        raise ValueError("Position should be between 0 and list length - 1.")

    # Insert value at the specified position
    lst.insert(position, value)

    # Remove the last value to maintain original length
    lst.pop()

    return lst

def remove_consecutive_duplicates(input_list):
    if not input_list:
        return [" "] * 5

    new_list = [input_list[0]]
    for i in range(1, len(input_list)):
        if input_list[i] != input_list[i - 1]:
            new_list.append(input_list[i])

    # Append empty strings if new_list length is less than 5
    while len(new_list) < 5:
        new_list.append(" ")

    return new_list

def main():
    test_file = "/kaggle/input/faiss-k5-l20/valid.csv"
    
     = "/kaggle/input/faiss-k5-l20/valid.csv"

    files_all = sorted(list(glob("/kaggle/input/cirruswiki-titles/*.parquet")))
    files_np = sorted(list(glob("/kaggle/input/enwiki-cirrus-20230701-e5-large-part*/*.npy")))
    files_all = [(x, y) for x, y in zip(files_all, files_np)]
    files = [files_all[: len(files_all) // 2], files_all[len(files_all) // 2 :]]

    model = SentenceTransformer("/kaggle/input/intfloat-e5-large-v2").to("cuda:0")

    test = pd.read_csv(test_file)
    embs = []
    for idx, row in test.iterrows():
        sentences = ["query: " + row.prompt + " " + row.A + " " + row.B + " " + row.C + " " + row.D + " " + row.E]
        embeddings = torch.Tensor(model.encode(sentences, show_progress_bar=False, normalize_embeddings=True))
        embs.append(torch.nn.functional.normalize(embeddings, dim=1))

    query_embeddings = torch.Tensor(np.stack(embs)).squeeze(1)

    TOP_K = 5

    all_vals_gpu_0 = torch.full((len(test), TOP_K), -float("inf"), dtype=torch.float16) 
    all_texts_gpu_0 = [[None] * TOP_K for _ in range(len(all_vals_gpu_0))]

    all_vals_gpu_1 = torch.full((len(test), TOP_K), -float("inf"), dtype=torch.float16)
    all_texts_gpu_1 = [[None] * TOP_K for _ in range(len(all_vals_gpu_1))]

    def load_data(files, device):
        for file, file_np in tqdm(files, total=len(files)):
            df = pd.read_parquet(file, engine="pyarrow", use_threads=True)
            file_embeddings = np.load(file_np)
            context_embeddings = torch.Tensor(file_embeddings).to(device).half()
            context_embeddings = torch.nn.functional.normalize(context_embeddings, dim=1)
            
            max_inds, max_vals = get_topk(
                    query_embeddings, context_embeddings, topk=TOP_K, bs=8
                )
            
            # loop through all queries (test)
            for i in range(len(test)):
                # start with highest new val (pos 0) vs worst value already in the toplist (pos topk - 1)
                for new in range(TOP_K):
                    if device == "cuda:0":
                        if max_vals[i][new].item() < all_vals_gpu_0[i][TOP_K - 1]:
                            break
                        for old in range(TOP_K):
                            if max_vals[i][new].item() > all_vals_gpu_0[i][old]:
                                all_vals_gpu_0[i] = insert_value_at(
                                    all_vals_gpu_0[i],
                                    value=max_vals[i][new].item(),
                                    position=old,
                                )
                                all_texts_gpu_0[i] = insert_value_at_list(
                                    all_texts_gpu_0[i],
                                    value=df.iloc[max_inds[i][new].item()].text,
                                    position=old,
                                )
                                break
                    else:
                        if max_vals[i][new].item() < all_vals_gpu_1[i][TOP_K - 1]:
                            break
                        for old in range(TOP_K):
                            if max_vals[i][new].item() > all_vals_gpu_1[i][old]:
                                all_vals_gpu_1[i] = insert_value_at(
                                    all_vals_gpu_1[i],
                                    value=max_vals[i][new].item(),
                                    position=old,
                                )
                                all_texts_gpu_1[i] = insert_value_at_list(
                                    all_texts_gpu_1[i],
                                    value=df.iloc[max_inds[i][new].item()].text,
                                    position=old,
                                )
                                break
                                
    Parallel(n_jobs=2, backend="threading")(
        delayed(load_data)(files[i], f"cuda:{i}") for i in range(2)
    )
    all_vals = torch.hstack([all_vals_gpu_0, all_vals_gpu_1])
    val, inds = torch.topk(all_vals.float(), axis=1, k=TOP_K)
    all_texts = [
        [(t0 + t1)[inner_idx.item()] for inner_idx in idx]
        for t0, t1, idx in zip(all_texts_gpu_0, all_texts_gpu_1, inds)
    ]
    all_texts = [remove_consecutive_duplicates(lst) for lst in all_texts]
    test["context"] = [
        "\n\n".join([x[i] for i in list(range(TOP_K))]) for x in all_texts
    ]
    test.to_csv("test_context_2.csv", index=False)


if __name__ == "__main__":
    main()