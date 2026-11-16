import numpy as np
import torch

from mmxai.graph.contains import ContainsGraph
from mmxai.models.subgraph_sage import (
    SubgraphSage,
    SubgraphTrainingData,
    candidate_subgraph_scores,
    explain_candidate,
    image_concept_matrix,
    masked_concept_mean,
    subgraph_batch_loss,
)
from mmxai.models.train import TrainingData, batch_loss

CONCEPT_X = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])


def test_image_concept_matrix_pads_with_minus_one():
    graph = ContainsGraph(
        image_ids=np.array([10, 11]),
        image_split=np.array(["train", "train"]),
        concept_keys=np.array(["a.n.01", "b.n.01", "c.n.01"]),
        edge_image=np.array([0, 0, 1]),
        edge_concept=np.array([0, 2, 1]),
    )
    assert image_concept_matrix(graph).tolist() == [[0, 2], [1, -1]]


def test_masked_mean_keeps_only_concepts_in_the_subgraph():
    image_concepts = torch.tensor([[[0, 2], [1, -1]]])
    subgraph = torch.tensor([[2, 1]])
    mean = masked_concept_mean(CONCEPT_X, image_concepts, subgraph)
    assert torch.allclose(mean[0, 0], CONCEPT_X[2])
    assert torch.allclose(mean[0, 1], CONCEPT_X[1])


def test_no_concept_in_the_subgraph_gives_zero():
    mean = masked_concept_mean(CONCEPT_X, torch.tensor([[[0, -1]]]), torch.tensor([[1, -1]]))
    assert torch.equal(mean, torch.zeros(1, 1, 2))


def test_padding_in_both_never_matches():
    mean = masked_concept_mean(CONCEPT_X, torch.tensor([[[-1, -1]]]), torch.tensor([[-1]]))
    assert torch.equal(mean, torch.zeros(1, 1, 2))


def test_untrained_model_returns_the_clip_vector():
    image_x = torch.tensor([[[0.3, 0.4]]])
    output = SubgraphSage(2)(image_x, torch.tensor([[[5.0, 5.0]]]))
    assert torch.allclose(output, image_x)


def tiny_training_data():
    return SubgraphTrainingData(
        image_x=torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]]),
        image_concepts=torch.tensor([[2, -1], [0, -1], [1, -1]]),
        concept_x=torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]),
        text_x=torch.tensor([[0.6, 0.8]]),
        text_image_row=torch.tensor([0]),
        candidate_rows=torch.tensor([[0, 1, 2]]),
        is_negative=torch.tensor([[False, True, True]]),
        text_subgraphs=torch.tensor([[2]]),
    )


def test_loss_is_finite_and_training_lowers_it():
    torch.manual_seed(0)
    data = tiny_training_data()
    model = SubgraphSage(2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    generator = torch.Generator().manual_seed(0)
    rows = torch.tensor([0])
    first = subgraph_batch_loss(model, data, rows, 2, 0.05, generator)
    for _ in range(30):
        loss = subgraph_batch_loss(model, data, rows, 2, 0.05, generator)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert torch.isfinite(first)
    assert loss.item() < first.item()


def test_untrained_loss_equals_the_stage_one_loss():
    data = tiny_training_data()
    # With W_nb = 0 the neighbour mean is ignored, so any values give the same loss.
    stage_one = TrainingData(
        image_x=data.image_x,
        neighbor_mean=torch.zeros(3, 2),
        text_x=data.text_x,
        text_image_row=data.text_image_row,
        candidate_rows=data.candidate_rows,
        is_negative=data.is_negative,
    )

    class CloneOfClip(SubgraphSage):
        def forward_from_neighbor_mean(self, image_x, neighbor_mean):
            return self.forward(image_x, neighbor_mean)

    rows = torch.tensor([0])
    model = CloneOfClip(2)
    new_loss = subgraph_batch_loss(model, data, rows, 2, 0.05, torch.Generator().manual_seed(3))
    old_loss = batch_loss(model, stage_one, rows, 2, 0.05, torch.Generator().manual_seed(3))
    assert torch.allclose(new_loss, old_loss)


def identity_neighbor_model():
    model = SubgraphSage(2)
    with torch.no_grad():
        model.neighbor_weight.weight.copy_(torch.eye(2))
    return model


def test_candidate_scores_match_a_manual_cosine():
    data = tiny_training_data()
    scores = candidate_subgraph_scores(
        identity_neighbor_model(),
        query_x=data.text_x,
        query_subgraphs=data.text_subgraphs,
        candidate_rows=torch.tensor([[0, 1]]),
        image_x=data.image_x,
        image_concepts=data.image_concepts,
        concept_x=data.concept_x,
        chunk=1,
    )
    h0 = data.image_x[0] + data.concept_x[2]
    h1 = data.image_x[1]
    q = data.text_x[0]
    expected = [
        torch.cosine_similarity(h0, q, dim=0).item(),
        torch.cosine_similarity(h1, q, dim=0).item(),
    ]
    assert np.allclose(scores[0], expected, atol=1e-6)


def test_chunk_size_does_not_change_the_scores():
    data = tiny_training_data()
    arguments = {
        "query_x": torch.tensor([[0.6, 0.8], [1.0, 0.0]]),
        "query_subgraphs": torch.tensor([[2], [0]]),
        "candidate_rows": torch.tensor([[0, 1, 2], [2, 1, 0]]),
        "image_x": data.image_x,
        "image_concepts": data.image_concepts,
        "concept_x": data.concept_x,
    }
    model = identity_neighbor_model()
    one = candidate_subgraph_scores(model, chunk=1, **arguments)
    two = candidate_subgraph_scores(model, chunk=2, **arguments)
    assert np.allclose(one, two, atol=1e-6)


def test_no_queries_gives_an_empty_score_table():
    data = tiny_training_data()
    scores = candidate_subgraph_scores(
        SubgraphSage(2),
        query_x=torch.zeros(0, 2),
        query_subgraphs=torch.zeros(0, 1, dtype=torch.long),
        candidate_rows=torch.zeros(0, 3, dtype=torch.long),
        image_x=data.image_x,
        image_concepts=data.image_concepts,
        concept_x=data.concept_x,
        chunk=2,
    )
    assert scores.shape == (0, 3)


def test_explanation_drops_each_matched_concept_and_recomputes():
    data = tiny_training_data()
    record = explain_candidate(
        identity_neighbor_model(),
        data.text_x[0],
        data.text_subgraphs[0],
        prized_rows={2},
        image_vector=data.image_x[0],
        image_concept_rows=data.image_concepts[0],
        concept_x=data.concept_x,
        concept_keys=np.array(["a.n.01", "b.n.01", "c.n.01"]),
    )
    assert [item["concept"] for item in record["matched"]] == ["c.n.01"]
    assert record["matched"][0]["bridge"] is False
    without = torch.cosine_similarity(data.image_x[0], data.text_x[0], dim=0).item()
    assert abs(record["matched"][0]["score_without"] - without) < 1e-6
