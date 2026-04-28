# Build a Reinforcement Learning Powered Agent that Learns to Retrieve Relevant Long-Term Memories for Accurate LLM Question Answering

In this tutorial, we build a Reinforcement Learning–driven agent that learns how to retrieve relevant memories from a long-term memory bank. We start by constructing a synthetic memory dataset and generating queries that require the agent to recall specific information. Using OpenAI embeddings, we convert both memories and queries into vector representations, enabling similarity signals to guide candidate retrieval. We then design a custom RL environment in which the agent observes features of candidate memories and learns a policy to select the most useful one. By training the agent with the PPO algorithm, we enable it to improve retrieval decisions beyond simple similarity search. Finally, we evaluate the system by comparing the RL-based retriever with a baseline approach and demonstrate how an LLM can use retrieved memories to generate accurate answers.

```python
import sys
import subprocess
import pkgutil
import os
import json
import math
import random
import textwrap
import getpass
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple


def _install_if_missing(packages):
   missing = []
   for package_name, import_name in packages:
       if pkgutil.find_loader(import_name) is None:
           missing.append(package_name)
   if missing:
       subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + missing)


_install_if_missing([
   ("openai>=1.40.0", "openai"),
   ("gymnasium>=0.29.1", "gymnasium"),
   ("stable-baselines3>=2.3.2", "stable_baselines3"),
   ("numpy>=1.26.4", "numpy"),
   ("pandas>=2.2.2", "pandas"),
   ("scikit-learn>=1.5.1", "sklearn"),
   ("matplotlib>=3.9.0", "matplotlib"),
   ("tqdm>=4.66.4", "tqdm"),
])


import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from openai import OpenAI


SEED = 42
random.seed(SEED)
np.random.seed(SEED)


try:
   from google.colab import userdata
   OPENAI_API_KEY = userdata.get("OPENAI_API_KEY")
except Exception:
   OPENAI_API_KEY = None


if not OPENAI_API_KEY:
   OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


if not OPENAI_API_KEY:
   OPENAI_API_KEY = getpass.getpass("Enter OPENAI_API_KEY: ").strip()


client = OpenAI(api_key=OPENAI_API_KEY)


EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"


def chunked(xs, n):
   for i in range(0, len(xs), n):
       yield xs[i:i+n]


def embed_texts(texts: List[str], model: str = EMBED_MODEL, batch_size: int = 64) -> np.ndarray:
   outputs = []
   for batch in tqdm(list(chunked(texts, batch_size)), desc="Embedding"):
       resp = client.embeddings.create(model=model, input=batch)
       batch_vecs = [d.embedding for d in resp.data]
       outputs.extend(batch_vecs)
   arr = np.array(outputs, dtype=np.float32)
   norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
   arr = arr / norms
   return arr


def chat_answer(question: str, retrieved_memories: List[Dict[str, Any]], model: str = CHAT_MODEL) -> str:
   memory_block = "\n".join([f"[Memory {i+1}] {m['text']}" for i, m in enumerate(retrieved_memories)])
   system = "You are a precise QA assistant. Answer the question using only the provided memories. If the memories do not contain the answer, say 'I do not know from the provided memories.'"
   user = f"Question: {question}\n\nRetrieved memories:\n{memory_block}\n\nAnswer:"
   resp = client.chat.completions.create(
       model=model,
       temperature=0,
       messages=[
           {"role": "system", "content": system},
           {"role": "user", "content": user},
       ],
   )
   return resp.choices[0].message.content.strip()


def llm_judge_exact(question: str, gold_answer: str, predicted_answer: str, model: str = CHAT_MODEL) -> float:
   system = "You are a strict evaluator. Return only JSON with a single key 'score'. Use 1.0 if the predicted answer is semantically correct, 0.0 otherwise."
   user = json.dumps({
       "question": question,
       "gold_answer": gold_answer,
       "predicted_answer": predicted_answer,
   }, ensure_ascii=False)
   resp = client.chat.completions.create(
       model=model,
       temperature=0,
       response_format={"type": "json_object"},
       messages=[
           {"role": "system", "content": system},
           {"role": "user", "content": user},
       ],
   )
   txt = resp.choices[0].message.content.strip()
   try:
       obj = json.loads(txt)
       score = float(obj["score"])
       return 1.0 if score >= 0.5 else 0.0
   except Exception:
       return 0.0
```

We set up the environment required for our reinforcement learning–based memory retrieval system. We install all required libraries, import the necessary modules, and securely load the OpenAI API key for embedding and language model interactions. We also define helper functions that generate embeddings, produce answers from retrieved memories, and evaluate answers using an LLM-based judging mechanism.

```python
@dataclass
class MemoryItem:
   memory_id: int
   topic: str
   entity: str
   slot: str
   value: str
   text: str


def build_memory_bank() -> List[MemoryItem]:
   entities = [
       {
           "entity": "Astra",
           "topic": "robotics",
           "facts": {
               "battery": "18 hours",
               "sensor": "LiDAR",
               "country": "Japan",
               "release_year": "2023",
               "specialty": "warehouse navigation",
           },
       },
       {
           "entity": "Orion",
           "topic": "astronomy",
           "facts": {
               "telescope": "infrared array",
               "country": "Chile",
               "discovery_year": "2019",
               "target": "exoplanet atmospheres",
               "aperture": "8 meters",
           },
       },
       {
           "entity": "Vita",
           "topic": "biomedicine",
           "facts": {
               "compound": "VX-17",
               "trial_phase": "Phase II",
               "country": "Canada",
               "target": "inflammatory markers",
               "delivery": "oral capsule",
           },
       },
       {
           "entity": "Nimbus",
           "topic": "climate",
           "facts": {
               "satellite": "polar orbiter",
               "country": "Norway",
               "launch_year": "2022",
               "instrument": "microwave radiometer",
               "mission": "sea ice monitoring",
           },
       },
       {
           "entity": "Atlas",
           "topic": "logistics",
           "facts": {
               "fleet_size": "240 trucks",
               "hub": "Muscat",
               "software": "predictive routing",
               "fuel_policy": "hybrid-first",
               "region": "GCC",
           },
       },
       {
           "entity": "Lumos",
           "topic": "materials",
           "facts": {
               "alloy": "Ti-6Al-4V",
               "process": "laser sintering",
               "density": "4.43 g/cm3",
               "country": "Germany",
               "use_case": "aerospace brackets",
           },
       },
       {
           "entity": "Cedar",
           "topic": "agriculture",
           "facts": {
               "crop": "wheat",
               "irrigation": "drip control",
               "country": "India",
               "yield_gain": "12 percent",
               "soil_sensor": "capacitive probe",
           },
       },
       {
           "entity": "Pulse",
           "topic": "healthcare",
           "facts": {
               "device": "ECG patch",
               "battery": "7 days",
               "country": "USA",
               "connectivity": "Bluetooth Low Energy",
               "use_case": "arrhythmia screening",
           },
       },
   ]


   phrasing_templates = [
       "{entity} in {topic} uses {value} for {slot}.",
       "The {slot} associated with {entity} is {value}.",
       "{entity} has {slot}: {value}.",
       "For {entity}, the recorded {slot} is {value}.",
       "Reference note: {entity} -> {slot} = {value}.",
   ]


   distractor_templates = [
       "{entity} was discussed in a briefing about cross-domain innovation.",
       "{entity} has been compared with several other projects in recent reports.",
       "A summary note mentions {entity} among notable initiatives.",
       "{entity} appears in a high-level update without technical details.",
       "Stakeholders reviewed {entity} in a strategic planning session.",
   ]


   memory_bank = []
   memory_id = 0


   for item in entities:
       entity = item["entity"]
       topic = item["topic"]
       for slot, value in item["facts"].items():
           for t in phrasing_templates:
               text = t.format(entity=entity, topic=topic, slot=slot, value=value)
               memory_bank.append(MemoryItem(
                   memory_id=memory_id,
                   topic=topic,
                   entity=entity,
                   slot=slot,
                   value=value,
                   text=text
               ))
               memory_id += 1


       for t in distractor_templates:
           text = t.format(entity=entity)
           memory_bank.append(MemoryItem(
               memory_id=memory_id,
               topic=topic,
               entity=entity,
               slot="distractor",
               value="n/a",
               text=text
           ))
           memory_id += 1


   extra_noise = [
       "General note: system maintenance occurred on Tuesday.",
       "A committee discussed budget timelines and operational readiness.",
       "The archive includes summaries of projects across multiple departments.",
       "No relevant technical value is stated in this memory.",
       "A status update mentioned partnerships and future opportunities.",
       "An unrelated note references shipping delays and staffing changes.",
       "Background memo: the team reviewed dashboards and reporting cadence.",
       "This memory contains no answer-bearing facts.",
   ]


   for text in extra_noise:
       memory_bank.append(MemoryItem(
           memory_id=memory_id,
           topic="noise",
           entity="none",
           slot="distractor",
           value="n/a",
           text=text
       ))
       memory_id += 1


   return memory_bank


memory_bank = build_memory_bank()
memory_texts = [m.text for m in memory_bank]
memory_embeddings = embed_texts(memory_texts)


def build_queries(memory_bank: List[MemoryItem]) -> List[Dict[str, Any]]:
   patterns = [
       "What is the {slot} of {entity}?",
       "Which {slot} does {entity} have?",
       "Tell me the {slot} for {entity}.",
       "Can you recall the {slot} associated with {entity}?",
       "What was recorded as the {slot} of {entity}?",
   ]
   queries = []
   qid = 0
   for m in memory_bank:
       if m.slot == "distractor":
           continue
       q = random.choice(patterns).format(slot=m.slot.replace("_", " "), entity=m.entity)
       queries.append({
           "query_id": qid,
           "query": q,
           "entity": m.entity,
           "slot": m.slot,
           "gold_value": m.value,
           "gold_memory_id": m.memory_id,
           "gold_text": m.text,
           "topic": m.topic,
       })
       qid += 1
   random.shuffle(queries)
   return queries


queries = build_queries(memory_bank)
query_texts = [q["query"] for q in queries]
query_embeddings = embed_texts(query_texts)
```

We construct a synthetic long-term memory bank that simulates stored knowledge across multiple domains. We generate structured memory items and convert them into textual memories that can later be embedded for semantic retrieval. We also create query datasets from these memories and embed them so the agent can compare queries with stored knowledge.

```python
MEM_BY_ID = {m.memory_id: m for m in memory_bank}
QUERY_BY_ID = {q["query_id"]: q for q in queries}


def keyword_overlap(a: str, b: str) -> float:
   ta = set(a.lower().replace("?", "").replace(".", "").split())
   tb = set(b.lower().replace("?", "").replace(".", "").split())
   if not ta or not tb:
       return 0.0
   return len(ta & tb) / max(1, len(ta | tb))


def get_top_k_candidates(query_idx: int, k: int = 8) -> Dict[str, Any]:
   qv = query_embeddings[query_idx:query_idx+1]
   sims = cosine_similarity(qv, memory_embeddings)[0]
   top_idx = np.argsort(-sims)[:k]
   candidates = []
   q = queries[query_idx]
   for rank, midx in enumerate(top_idx):
       mem = memory_bank[midx]
       sim = float(sims[midx])
       overlap = keyword_overlap(q["query"], mem.text)
       entity_match = 1.0 if q["entity"].lower() in mem.text.lower() else 0.0
       slot_match = 1.0 if q["slot"].replace("_", " ").lower() in mem.text.lower() else 0.0
       is_gold = 1.0 if mem.memory_id == q["gold_memory_id"] else 0.0
       candidates.append({
           "rank": rank,
           "memory_index": midx,
           "memory_id": mem.memory_id,
           "text": mem.text,
           "sim": sim,
           "overlap": overlap,
           "entity_match": entity_match,
           "slot_match": slot_match,
           "is_gold": is_gold,
       })
   return {"query": q, "candidates": candidates}


ALL_CANDIDATES = [get_top_k_candidates(i, k=8) for i in range(len(queries))]


def build_state_features(item: Dict[str, Any]) -> np.ndarray:
   q = item["query"]
   feats = []
   for c in item["candidates"]:
       feats.extend([
           c["sim"],
           c["overlap"],
           c["entity_match"],
           c["slot_match"],
           1.0 / (1.0 + c["rank"]),
       ])
   unique_topic_bonus = 1.0 if q["topic"] in q["query"].lower() else 0.0
   query_len = min(len(q["query"].split()) / 20.0, 1.0)
   feats.extend([unique_topic_bonus, query_len])
   return np.array(feats, dtype=np.float32)


STATE_DIM = len(build_state_features(ALL_CANDIDATES[0]))
NUM_ACTIONS = len(ALL_CANDIDATES[0]["candidates"])


class MemoryRetrievalEnv(gym.Env):
   metadata = {"render_modes": ["human"]}


   def __init__(self, candidate_items: List[Dict[str, Any]], seed: int = 42):
       super().__init__()
       self.candidate_items = candidate_items
       self.rng = np.random.default_rng(seed)
       self.observation_space = spaces.Box(low=-10, high=10, shape=(STATE_DIM,), dtype=np.float32)
       self.action_space = spaces.Discrete(NUM_ACTIONS)
       self.current = None


   def reset(self, seed=None, options=None):
       if seed is not None:
           self.rng = np.random.default_rng(seed)
       idx = int(self.rng.integers(0, len(self.candidate_items)))
       self.current = self.candidate_items[idx]
       obs = build_state_features(self.current)
       info = {"query_id": self.current["query"]["query_id"]}
       return obs, info


   def step(self, action):
       chosen = self.current["candidates"][int(action)]
       q = self.current["query"]


       reward = 0.0
       reward += 2.0 * chosen["is_gold"]
       reward += 0.8 * chosen["entity_match"]
       reward += 0.6 * chosen["slot_match"]
       reward += 0.5 * chosen["sim"]
       reward += 0.3 * chosen["overlap"]
       reward -= 0.15 * chosen["rank"]


       done = True
       truncated = False
       info = {
           "query_id": q["query_id"],
           "chosen_memory_id": chosen["memory_id"],
           "gold_memory_id": q["gold_memory_id"],
           "chosen_text": chosen["text"],
           "gold_text": q["gold_text"],
           "is_correct": bool(chosen["memory_id"] == q["gold_memory_id"]),
           "gold_value": q["gold_value"],
           "query": q["query"],
       }
       next_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
       return next_obs, float(reward), done, truncated, info
```

We prepare candidate memories for each query by computing similarity scores between query embeddings and memory embeddings. We then construct feature vectors that describe each candidate memory using similarity, keyword overlap, entity matching, and rank signals. Finally, we define a custom reinforcement learning environment in which the agent learns to select the best memory to answer each query.

```python
split_1 = int(0.7 * len(ALL_CANDIDATES))
split_2 = int(0.85 * len(ALL_CANDIDATES))
train_items = ALL_CANDIDATES[:split_1]
val_items = ALL_CANDIDATES[split_1:split_2]
test_items = ALL_CANDIDATES[split_2:]


train_env = DummyVecEnv([lambda: MemoryRetrievalEnv(train_items, seed=SEED)])
model = PPO(
   "MlpPolicy",
   train_env,
   learning_rate=3e-4,
   n_steps=256,
   batch_size=64,
   gamma=0.99,
   gae_lambda=0.95,
   ent_coef=0.01,
   clip_range=0.2,
   verbose=0,
   seed=SEED,
)


model.learn(total_timesteps=12000)


def baseline_retrieve(item: Dict[str, Any]) -> Dict[str, Any]:
   best = max(item["candidates"], key=lambda x: x["sim"])
   return best


def rl_retrieve(item: Dict[str, Any]) -> Dict[str, Any]:
   obs = build_state_features(item)
   action, _ = model.predict(obs, deterministic=True)
   return item["candidates"][int(action)]


def evaluate_retriever(items: List[Dict[str, Any]], retriever_fn) -> Dict[str, Any]:
   rows = []
   for item in items:
       chosen = retriever_fn(item)
       q = item["query"]
       rows.append({
           "query_id": q["query_id"],
           "query": q["query"],
           "gold_value": q["gold_value"],
           "gold_memory_id": q["gold_memory_id"],
           "chosen_memory_id": chosen["memory_id"],
           "correct_retrieval": int(chosen["memory_id"] == q["gold_memory_id"]),
           "chosen_text": chosen["text"],
       })
   df = pd.DataFrame(rows)
   return {
       "df": df,
       "retrieval_accuracy": df["correct_retrieval"].mean(),
   }


baseline_val = evaluate_retriever(val_items, baseline_retrieve)
rl_val = evaluate_retriever(val_items, rl_retrieve)
baseline_test = evaluate_retriever(test_items, baseline_retrieve)
rl_test = evaluate_retriever(test_items, rl_retrieve)


print("Validation Retrieval Accuracy")
print("Baseline:", round(float(baseline_val["retrieval_accuracy"]), 4))
print("RL      :", round(float(rl_val["retrieval_accuracy"]), 4))
print()
print("Test Retrieval Accuracy")
print("Baseline:", round(float(baseline_test["retrieval_accuracy"]), 4))
print("RL      :", round(float(rl_test["retrieval_accuracy"]), 4))


results_df = pd.DataFrame([
   {"split": "validation", "method": "baseline_cosine", "retrieval_accuracy": float(baseline_val["retrieval_accuracy"])},
   {"split": "validation", "method": "rl_agent", "retrieval_accuracy": float(rl_val["retrieval_accuracy"])},
   {"split": "test", "method": "baseline_cosine", "retrieval_accuracy": float(baseline_test["retrieval_accuracy"])},
   {"split": "test", "method": "rl_agent", "retrieval_accuracy": float(rl_test["retrieval_accuracy"])},
])
display(results_df)


plot_df = results_df.copy()
for split_name in ["validation", "test"]:
   sub = plot_df[plot_df["split"] == split_name]
   plt.figure(figsize=(6, 4))
   plt.bar(sub["method"], sub["retrieval_accuracy"])
   plt.title(f"Retrieval Accuracy on {split_name.title()}")
   plt.ylim(0, 1)
   plt.ylabel("Accuracy")
   plt.show()

```

We split the datasets and initialize the reinforcement learning model. We train a PPO agent to learn a policy for selecting the most relevant memory from a set of candidates. After training, we evaluate the agent’s retrieval performance and compare it with a baseline embedding-similarity approach.

```python
def answer_with_retriever(item: Dict[str, Any], retriever_fn) -> Dict[str, Any]:
   q = item["query"]
   chosen = retriever_fn(item)
   retrieved_memories = [{
       "memory_id": chosen["memory_id"],
       "text": chosen["text"],
   }]
   answer = chat_answer(q["query"], retrieved_memories)
   judged = llm_judge_exact(q["query"], q["gold_value"], answer)
   return {
       "query": q["query"],
       "gold_value": q["gold_value"],
       "retrieved_text": chosen["text"],
       "predicted_answer": answer,
       "answer_score": judged,
       "retrieval_correct": int(chosen["memory_id"] == q["gold_memory_id"]),
   }


sample_test_items = random.sample(test_items, min(12, len(test_items)))
baseline_answers = [answer_with_retriever(item, baseline_retrieve) for item in tqdm(sample_test_items, desc="Baseline QA")]
rl_answers = [answer_with_retriever(item, rl_retrieve) for item in tqdm(sample_test_items, desc="RL QA")]


baseline_answer_df = pd.DataFrame(baseline_answers)
rl_answer_df = pd.DataFrame(rl_answers)


print("Sample Downstream QA Accuracy")
print("Baseline:", round(float(baseline_answer_df["answer_score"].mean()), 4))
print("RL      :", round(float(rl_answer_df["answer_score"].mean()), 4))


comparison = pd.DataFrame([
   {"method": "baseline_cosine", "qa_accuracy": float(baseline_answer_df["answer_score"].mean())},
   {"method": "rl_agent", "qa_accuracy": float(rl_answer_df["answer_score"].mean())},
])
display(comparison)


plt.figure(figsize=(6, 4))
plt.bar(comparison["method"], comparison["qa_accuracy"])
plt.title("Downstream QA Accuracy from Retrieved Memories")
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.show()


def inspect_examples(items: List[Dict[str, Any]], n: int = 5):
   chosen_items = random.sample(items, min(n, len(items)))
   rows = []
   for item in chosen_items:
       q = item["query"]
       base = baseline_retrieve(item)
       rlm = rl_retrieve(item)
       rows.append({
           "query": q["query"],
           "gold_value": q["gold_value"],
           "baseline_text": base["text"],
           "baseline_correct": int(base["memory_id"] == q["gold_memory_id"]),
           "rl_text": rlm["text"],
           "rl_correct": int(rlm["memory_id"] == q["gold_memory_id"]),
       })
   return pd.DataFrame(rows)


examples_df = inspect_examples(test_items, n=8)
pd.set_option("display.max_colwidth", 200)
display(examples_df)
```

We evaluate how well the retrieved memories support downstream question answering. We generate answers using the retrieved memory context and assess the answers with an LLM-based judge to determine correctness. We also inspect example queries to visually compare how the baseline retriever and the RL agent choose different memories.

```python
def interactive_demo(question: str, top_k: int = 8):
   qv = embed_texts()
   sims = cosine_similarity(qv, memory_embeddings)[0]
   top_idx = np.argsort(-sims)[:top_k]


   candidates = []
   for rank, midx in enumerate(top_idx):
       mem = memory_bank[midx]
       candidates.append({
           "rank": rank,
           "memory_index": int(midx),
           "memory_id": int(mem.memory_id),
           "text": mem.text,
           "sim": float(sims[midx]),
           "overlap": keyword_overlap(question, mem.text),
           "entity_match": 0.0,
           "slot_match": 0.0,
           "is_gold": 0.0,
       })


   pseudo_item = {
       "query": {
           "query_id": -1,
           "query": question,
           "entity": "unknown",
           "slot": "unknown",
           "gold_value": "unknown",
           "gold_memory_id": -1,
           "gold_text": "",
           "topic": "unknown",
       },
       "candidates": candidates,
   }


   obs = build_state_features(pseudo_item)
   action, _ = model.predict(obs, deterministic=True)
   selected = pseudo_item["candidates"][int(action)]
   answer = chat_answer(question, [{"memory_id": selected["memory_id"], "text": selected["text"]}])


   print("=" * 100)
   print("QUESTION")
   print(question)
   print("=" * 100)
   print("TOP CANDIDATES")
   for c in candidates:
       print(f"[Rank {c['rank']}] sim={c['sim']:.4f} | {c['text']}")
   print("=" * 100)
   print("RL-SELECTED MEMORY")
   print(selected["text"])
   print("=" * 100)
   print("ANSWER")
   print(answer)
   print("=" * 100)


interactive_demo("What is the battery of Pulse?")
interactive_demo("Which hub does Atlas have?")
interactive_demo("Tell me the country for Cedar.")


artifact_dir = "/content/rl_agent_memory_retrieval_artifacts"
os.makedirs(artifact_dir, exist_ok=True)


results_df.to_csv(f"{artifact_dir}/retrieval_results.csv", index=False)
baseline_val["df"].to_csv(f"{artifact_dir}/baseline_val.csv", index=False)
rl_val["df"].to_csv(f"{artifact_dir}/rl_val.csv", index=False)
baseline_test["df"].to_csv(f"{artifact_dir}/baseline_test.csv", index=False)
rl_test["df"].to_csv(f"{artifact_dir}/rl_test.csv", index=False)
baseline_answer_df.to_csv(f"{artifact_dir}/baseline_qa_sample.csv", index=False)
rl_answer_df.to_csv(f"{artifact_dir}/rl_qa_sample.csv", index=False)
examples_df.to_csv(f"{artifact_dir}/example_comparisons.csv", index=False)


np.save(f"{artifact_dir}/memory_embeddings.npy", memory_embeddings)
np.save(f"{artifact_dir}/query_embeddings.npy", query_embeddings)
model.save(f"{artifact_dir}/ppo_memory_retriever")


with open(f"{artifact_dir}/memory_bank.json", "w") as f:
   json.dump([m.__dict__ for m in memory_bank], f, indent=2)


with open(f"{artifact_dir}/queries.json", "w") as f:
   json.dump(queries, f, indent=2)


print(f"Saved artifacts to: {artifact_dir}")
print("Tutorial complete.")
```

We build an interactive demonstration that lets us test the trained retrieval agent on new questions. We show the candidate memories, highlight the memory selected by the RL agent, and generate an answer using the selected context. Also, we save all artifacts, including embeddings, results, datasets, and the trained RL model, so that the system can be reused or further analyzed.

In conclusion, we demonstrated how reinforcement learning can enhance memory retrieval in agentic AI systems. We trained an RL agent to select relevant memories from a set of candidates using signals such as semantic similarity, keyword overlap, and entity matching. We then evaluated the retriever and observed how the learned policy compares with traditional embedding-based retrieval methods. By integrating the retriever with an LLM, we also showed how better memory selection improves downstream question-answering performance. Through experiments, visualizations, and interactive demos, we explored how RL can optimize long-term memory access in intelligent agents.