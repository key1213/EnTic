central_entity_prompt = """

Given a natural language question, identify and extract the central named entities that are essential for answering the question.
These may include people, organizations, locations, objects, works (books, movies, music), etc.

Here are six examples:

Q: Lou Seal is the mascot for the team that last won the World Series when?
The output is:
{{"entities": ["Lou Seal"]}}

Q: Which movie did Kim Richards appear in and had a length of 112.0 minutes?
The output is:
{{"entities": ["Kim Richards"]}}

Q: To whom was the actor who was in the movie "Avatar" married in 2012?
The output is:
{{"entities": ["Avatar"]}}

Q: What is the capital of the country where the Eiffel Tower is located?
The output is:
{{"entities": ["Eiffel Tower"]}}

Q: What religion does the leader Ovadia Yosef follow?
The output is:
{{"entities": ["Ovadia Yosef"]}}

Q: Where did the artist of the 'Country Nation World Tour' concert go to college?
The output is:
{{"entities": ["Country Nation World Tour"]}}

Now extract the central entities for the following question in JSON format.
Q: {question}
You output one single line only, with no explanations, no code fences, no leading text. The line must be exactly in this format:
{{"entities": ["..."]}}
If nothing is found, output:
{{"entities": []}}

"""


# ----------------------------------
# EnTic: Reasoning Module - Thinking Stage
# ----------------------------------
formulate_subtask_prompt = """Given the original question and the current state of the reasoning process (current path, previous action, previous observation), formulate the next subtask to guide exploration in the knowledge graph. Represent entities using their names when possible, but entity IDs are acceptable in observation. Format your output as a JSON object with a single key: "subtask".
Avoid exploring type-level entities (e.g., TV Episode, Language, Event Class). Focus on instance-level entities that are directly involved in answering the question.
Always ensure the subtask focuses on the current entity and determines the immediate next step of reasoning starting from this entity to advance towards the overall question's answer.

Here are several examples:

Q: Name the president of the country whose main spoken language was Brahui in 1980?
Current State:
Current Path: ["Brahui Language"]
Previous Action: None
Previous Observation: None
Current Entity: Brahui Language
Current Entity ID: m.012345
The output in JSON format is:
{{"subtask": "Find the country where Brahui is the main spoken language."}}

Q: Name the president of the country whose main spoken language was Brahui in 1980?
Current State:
Current Path: [["Brahui Language", "language.human_language.main_country", "Pakistan"]]
Previous Action: "Select country relation from Brahui Language"
Previous Observation:{{"language.human_language.main_country": ["m.06789"]}}
Current Entity: Pakistan
Current Entity ID: m.06789
The output in JSON format is:
{{"subtask": "Find the president of Pakistan in 1980."}}


Q: Lou Seal is the mascot for the team that last won the World Series when?
Current State:
Current Path: ["Lou Seal"]
Previous Action: None
Previous Observation: None
Current Entity: Lou Seal
Current Entity ID: m.03_dwn
The output in JSON format is:
{{"subtask": "Find the team for which Lou Seal is the mascot."}}


Q: Lou Seal is the mascot for the team that last won the World Series when?
Current State:
Current Path: [["Lou Seal", "sports.mascot.team", "San Francisco Giants"]]
Previous Action: "Find mascot team"
Previous Observation:{{"sports.mascot.team": ["m.0713r"]}}
Current Entity: San Francisco Giants
Current Entity ID: m.0713r
The output in JSON format is:
{{"subtask": "Find the year when the San Francisco Giants last won the World Series."}}


Q: What city is the University of Oxford located in?
Current State:
Current Path: ["University of Oxford"]
Previous Action: None
Previous Observation: None
Current Entity: University of Oxford
Current Entity ID: m.oxford_u
The output in JSON format is:
{{"subtask": "Find the city where the University of Oxford is located."}}

Now formulate the next subtask for the following inputs in JSON format (must include "subtask" key). Output only the JSON.
Q: {question}
Current State:
Current Path: {current_path}
Previous Action: {previous_action}
Previous Observation: {previous_observation}
Current Entity: {current_entity_name}
Current Entity ID: {current_entity_id}
You output one single line only, with no explanations, no code fences, no leading text. The line must be exactly in this format:
{{"subtask": "..."}}
"""


# ----------------------------------
# EnTic: Reasoning Module - Selection Stage
# ----------------------------------
select_relevant_relations_prompt = """Based on the question and current subtask, rate each relation from the 'Possible Relations' list by how well it helps complete the subtask and overall question. All scores must be between 0 and 1 and sum to exactly 1.
Please return relation names in standard Freebase format (e.g., sports.mascot.team) rather than readable labels like sports mascot team.

Here are several examples to guide your scoring and selection:
Q: Name the president of the country whose main spoken language was Brahui in 1980?
Subtask: 'Find the country where Brahui is the main spoken language.'
Topic Entity: Brahui Language
Possible Relations: language.human_language.main_country; language.human_language.language_family; language.human_language.iso_639_3_code; base.rosetta.languoid.parent; language.human_language.writing_system; base.rosetta.languoid.languoid_class; language.human_language.countries_spoken_in; kg.object_profile.prominent_type
Output:
{{
   "relation_scores": [
      ["language.human_language.main_country", 0.45],
      ["language.human_language.countries_spoken_in", 0.35],
      ["base.rosetta.languoid.parent", 0.20]
   ],
   "selected_relations": ["language.human_language.main_country", "language.human_language.countries_spoken_in", "base.rosetta.languoid.parent"]
}}


Q: Who is the mascot of the team that last won the World Series?
Subtask: 'Find the team for which Lou Seal is the mascot.'
Topic Entity: Lou Seal
Possible Relations: sports.mascot.team; sports.sports_team.championships; sports.sports_team.last_championship; common.topic.notable_for; type.object.name; type.object.type
Output:
{{
   "relation_scores": [
      ["sports.mascot.team", 0.5],
      ["sports.sports_team.last_championship", 0.3],
      ["sports.sports_team.championships", 0.2]
   ],
   "selected_relations": ["sports.mascot.team", "sports.sports_team.last_championship", "sports.sports_team.championships"]
}}

Now score and select top-{k} relevant relations for the following case in JSON format. The output must include both "relation_scores" and "selected_relations" keys, and all scores must sum to exactly 1. Output only the JSON object.
Q: {question}
Subtask: {subtask}
Topic Entity: {topic_entity}
Possible Relations: {possible_relations}
You output with no explanations, no code fences, no leading text. The output must be exactly in this format:
{{
   "relation_scores": [
      ...
   ],
   "selected_relations": ["..."]
}}
"""


# -----------------------------
# EnTic: Answer Generation
# -----------------------------
answer_prompt_origin = """
Given the question and the reasoning paths (represented using entity names) that led to the answer, generate a clear and concise final answer.

Here is an example:
Q: Who is the coach of the team owned by Steve Bisciotti?
Reasoning paths: [
    [("Steve Bisciotti", "teams owned", "Baltimore Ravens"), ("Baltimore Ravens", "head coach", "John Harbaugh")]
] 

The output in JSON format is:
{{"answer": "John Harbaugh is the coach of the Baltimore Ravens, which is owned by Steve Bisciotti."}}

Now generate the answer for the following question in JSON format (must include "answer" key). Output only the JSON.
Q: {question}
Reasoning paths: {reasoning_paths} # Path using entity names
"""

answer_prompt_with_subtask= """
Given the question and the reasoning paths (represented using entity names) that led to the answer, and the current subtask that guides the exploration, generate a clear and concise final answer.

The answer should:
- Clearly explain the reasoning trace if it's multi-hop.
- Be factually grounded in the given path.
- Avoid hallucination or adding extra facts not supported by the path.
- Be output strictly in JSON format with a single key: "answer".
- If no answer can be found based on the reasoning paths or subtask,  Output: {{"answer":"Could not find an answer based on the reasoning paths."}}

Here are some examples:

Q: Who is the coach of the team owned by Steve Bisciotti?  
Reasoning paths: [
    [("Steve Bisciotti", "teams owned", "Baltimore Ravens"), ("Baltimore Ravens", "head coach", "John Harbaugh")]
]  
Subtask: "Find the coach of the Baltimore Ravens."
Output:  
{{"answer": "John Harbaugh is the coach of the Baltimore Ravens, which is owned by Steve Bisciotti."}}

Q: What city is the University of Oxford located in?  
Reasoning paths: [
    [("University of Oxford", "located in", "Oxford")]
]  
Subtask: "Find the city where the University of Oxford is located."
Output:  
{{"answer": "The University of Oxford is located in Oxford."}}

Q: Who is the spouse of the actor in the movie 'Titanic'?  
Reasoning paths: [
    [("Titanic", "featured actor", "Leonardo DiCaprio"), ("Leonardo DiCaprio", "spouse", "None")]
]  
Subtask: "Find the marital status of Leonardo DiCaprio."
Output:  
{{"answer": "Leonardo DiCaprio, an actor in Titanic, is not married."}}

Q: Who directed the movie that won Best Picture at the 1980 Academy Awards?  
Reasoning paths: [
    [("Academy Award for Best Picture (1980)", "won by", "Ordinary People"), ("Ordinary People", "directed by", "Robert Redford")]
]  
Subtask: "Find the director of 'Ordinary People'."
Output:  
{{"answer": "Robert Redford directed 'Ordinary People', which won Best Picture at the 1980 Academy Awards."}}

Q: Where did the 'Country Nation World Tour' concert artist go to college?  
Reasoning paths: [
    [("Country Nation World Tour", "music.concert_tour.artist", "Brad Paisley"), ("Brad Paisley", "educated at", "Belmont University")]
]  
Subtask: "Find the college that Brad Paisley attended."
Output:  
{{"answer": "Brad Paisley, the artist of the Country Nation World Tour, went to Belmont University."}}

Now generate the answer for the following question in JSON format (must include "answer" key). Output only the JSON.

Q: {question}  
Reasoning paths: {reasoning_paths}  
Subtask: {subtask}
"""

answer_prompt= """
Given the question and the reasoning paths (represented using entity names) that led to the answer, generate a clear and concise final answer.

The answer should:
- Clearly explain the reasoning trace if it's multi-hop.
- Be factually grounded in the given path.
- Avoid hallucination or adding extra facts not supported by the path.
- Be output strictly in JSON format with a single key: "answer".
- If no answer can be found based on the reasoning paths, Output: {{"answer":"Could not find an answer based on the reasoning paths."}}

Here are some examples:

Q: Who is the coach of the team owned by Steve Bisciotti?  
Reasoning paths: [
    [("Steve Bisciotti", "teams owned", "Baltimore Ravens"), ("Baltimore Ravens", "head coach", "John Harbaugh")]
]  
Output:  
{{"answer": "John Harbaugh is the coach of the Baltimore Ravens, which is owned by Steve Bisciotti."}}

Q: What city is the University of Oxford located in?  
Reasoning paths: [
    [("University of Oxford", "located in", "Oxford")]
]  
Output:  
{{"answer": "The University of Oxford is located in Oxford."}}

Q: Who is the spouse of the actor in the movie 'Titanic'?  
Reasoning paths: [
    [("Titanic", "featured actor", "Leonardo DiCaprio"), ("Leonardo DiCaprio", "spouse", "None")]
]  
Output:  
{{"answer": "Leonardo DiCaprio, an actor in Titanic, is not married."}}

Q: Who directed the movie that won Best Picture at the 1980 Academy Awards?  
Reasoning paths: [
    [("Academy Award for Best Picture (1980)", "won by", "Ordinary People"), ("Ordinary People", "directed by", "Robert Redford")]
]  
Output:  
{{"answer": "Robert Redford directed 'Ordinary People', which won Best Picture at the 1980 Academy Awards."}}

Q: Where did the 'Country Nation World Tour' concert artist go to college?  
Reasoning paths: [
    [("Country Nation World Tour", "music.concert_tour.artist", "Brad Paisley"), ("Brad Paisley", "educated at", "Belmont University")]
]  
Output:  
{{"answer": "Brad Paisley, the artist of the Country Nation World Tour, went to Belmont University."}}

Now generate the answer for the following question in JSON format (must include "answer" key). Output only the JSON.

Q: {question}  
Reasoning paths: {reasoning_paths}
You output one single line only, with no explanations, no code fences, no leading text. If the final answer cannot be found from the provided reasoning paths, return:
{{"answer":"Could not find an answer based on the reasoning paths."}}
Do not answer from background knowledge alone. The line must be exactly in this format:
{{"answer": "..."}}
"""


# ----------------------------------

# EnTic: Termination Module

# ----------------------------------

termination_prompt = """
Given the question and the current reasoning state, determine if enough information has been gathered to answer the question or if reasoning should continue.
Current state includes the reasoning path and latest observations.

Here are three examples:

Q: Name the president of the country whose main spoken language was Brahui in 1980?
Current state:
Path: [["Brahui Language", "language.human_language.main_country", "Pakistan"]]
Observation: {{
  "language.human_language.main_country": ["m.06789"]
}}

Output:
{{
  "termination": "no",
  "reason": "Need to find the president of the country whose main spoken language was Brahui in 1980.",
  "confidence": 0.65
}}


Q: Who is the coach of the team owned by Steve Bisciotti?
Current state:
Path: [["Steve Bisciotti", "teams.owned", "Baltimore Ravens"], ["Baltimore Ravens", "american_football.football_team.current_head_coach", "John Harbaugh"]]
Observation: {{
  "american_football.football_team.current_head_coach": ["m.12345"],
  "sports.sports_team.championships": ["m.2000", "m.2012"],
  "sports.sports_team.team_mascot": ["m.03_dwn"]

}}

Output:
{{
  "termination": "yes",
  "reason": "The relation \"current_head_coach\" has led to the direct answer: John Harbaugh.",
  "confidence": 0.9
}}



Now evaluate the following case. Consider:
1. Is the current entity a potential intermediate step in a multi-hop reasoning path?
2. Does the current path show progress towards answering the question?
3. Have we reached a dead end (no more relevant relations to explore)?
4. Have we found the final answer to the question?
5. If the question involves finding the latest, earliest, or any kind of maximum/minimum (e.g., "last won", "most recent", "earliest time"), DO NOT terminate unless you have found and compared ALL relevant candidate entities. Only terminate when you are certain the final answer has the correct extremal value.

You output with no explanations, no code fences, no leading text. Respond with a JSON object containing:

{{

  "termination": "yes" or "no",

  "reason": "Detailed explanation of why to continue or stop",

  "confidence": 0.0 to 1.0

}}


Q: {question}

Current state: {current_state}

"""

evaluation_prompt = """
You are given a natural language question and a subtask derived during multi-hop reasoning over a knowledge graph. 
Evaluate how well this subtask contributes to answering the original question.

Scoring Criteria:
- Score 1.0: The subtask directly and fully aligns with the question’s intent.
- Score between 0 and 1 (e.g., 0.5): The subtask is partially helpful or relevant but incomplete or too general.
- Score 0: The subtask is no progress.
- Score between -1 and 0: The subtask is misleading, unrelated, or incorrect.

Here are some examples:

Q: What is the capital of the country where the Eiffel Tower is located?  
Subtask: Find the country where the Eiffel Tower is located.  
Output:  
{{"score": 1.0}}

Q: Who is the spouse of the actor in Avatar?  
Subtask: Find the director of the movie Avatar.  
Output:  
{{"score": -1.0}}

Q: Where did the artist of the 'Country Nation World Tour' go to college?  
Subtask: Find the artist who performed the 'Country Nation World Tour'.  
Output:  
{{"score": 1.0}}

Q: Which city is the University of Oxford located in?  
Subtask: List all universities in Oxford.  
Output:  
{{"score": 0.3}}

Now evaluate the following case,respond with a JSON object containing:

Q: {question}  

Subtask: {subtask} 
You output one single line only, with no explanations, no code fences, no leading text. The line must be exactly in this format:
{{"score": ...}}
"""

estimate_optimal_path_prompt = """
You are given a natural language question that requires multi-hop reasoning over a knowledge graph. 
Based on the structure and complexity of the question, estimate the optimal number of reasoning steps (path length) needed to reach the final answer.

Guidelines:
- A simple lookup question typically needs 1 hop.
- A question involving an entity and one relation (e.g., birthplace, capital) usually takes 1–2 hops.
- If the question requires indirect reasoning (e.g., actor → movie → director → spouse), it may need 3–4 hops.
- Very complex or compositional questions may require up to 5 hops.

Only return a single integer between 1 and 5, indicating the estimated optimal number of hops in the reasoning path.

Here are some examples:

Q: What city is the University of Oxford located in?  
Output:  
{{"answer": 1}}

Q: Who is the spouse of the actor in the movie Titanic?  
Output:  
{{"answer": 3}}

Q: What is the capital of the country where the Eiffel Tower is located?  
Output:  
{{"answer": 2}}

Q: Who coached the team owned by Steve Bisciotti when they won the Super Bowl?  
Output:  
{{"answer": 4}}

Now evaluate the following case,respond with a JSON object containing:

Q: {question}  
You output one single line only, with no explanations, no code fences, no leading text. The line must be exactly in this format:
{{"answer": ...}}
"""
