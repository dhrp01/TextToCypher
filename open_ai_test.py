from openai import OpenAI

client = OpenAI(
  api_key="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"
)

prompt = """This is an example node for a graph database Node properties:
 - **Movie** - title: STRING Example: "The Matrix" - votes: INTEGER Min: 1, 
 Max: 5259 - tagline: STRING Example: "Welcome to the Real World" - released: 
 INTEGER Min: 1975, Max: 2012 - **Person** - born: INTEGER Min: 1929, 
 Max: 1996 - name: STRING Example: "Keanu Reeves" Relationship properties: - **ACTED_IN** - roles: 
 LIST Min Size: 1, Max Size: 6 - **REVIEWED** - summary: STRING Available options: 
 ['Pretty funny at times', 'A solid romp', 'Silly, but fun', 'You had me at Jerry', 'An amazing journey', 'Slapstick redeemed only by the Robin Williams and ', 'Dark, but compelling', 'The coolest football movie ever', 'Fun, but a little far fetched']
   - rating: INTEGER Min: 45, Max: 100 The relationships: (:Person)-[:ACTED_IN]->(:Movie) (:Person)-[:DIRECTED]->(:Movie) (:Person)-[:PRODUCED]->(:Movie)
     (:Person)-[:WROTE]->(:Movie) (:Person)-[:FOLLOWS]->(:Person) 
     (:Person)-[:REVIEWED]->(:Movie) Can you generate queries to answer the following questions:"""

questions = ["How many movies has Keanu Reeves acted in?",
    "How old is Keanu Reeves?",
    "How many movies have Keanu Reeves and Tom Hanks acted in together?",
    "How many movies have had the exact same number of actors act in them?",
    "What actors have never been in a movie with Charlize Theron?",
    "How many actors have been in less than 3 movies?"
]

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[
    {"role": "user", "content": prompt+questions[0]}
  ]
)

print(completion.choices[0].message);
