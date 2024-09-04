prompt_templates = {
    "GSM_conf": f"Here is a math problem and some steps:\n hhx \nPlease give your confidence.",
    "GSM_format":f"Here is a math probem:\n hhx \nPlease think step by step",
    "CSQA_conf": f"Below is a question and some steps:\nQuestion:\n{{question}}Please give your confidence.",
    "CSQA_format":f"Below is a question, analyze step by step and select one correct answer from it:\nQuestion:\n{{question}}",
    "TriviaQA_conf": f"Below is a question and some steps:\nQuestion:\n{{question}}Please give your confidence.",
    "TriviaQA_format":f"Below is a question, analyze step by step and select one correct answer from it:\nQuestion:\n{{question}}",
    "CSQA_Multistep":"""Read the question, break down the problem into K steps, think step by step, give your confidence in each step, and then derive your final answer and your confidence in this answer. 
Note: The confidence indicates how likely you think your answer is true.\n
Use the following format to answer:\n\n
Step 1: [Your reasoning], Confidence: [ONLY the confidence value that this step is correct]%\n...\n
Step K: [Your reasoning], Confidence: [ONLY the confidence value that this step is correct]%\n
Final Answer: [ONLY the {answer_type}; not a complete sentence]
Overall Confidence(0-100): [Your confidence value]%

Please refer to the example I have given:
<example>
Question:
They made a mess eating on the sofa bed, they dripped a tangy what sauce all over it?
A. basement
B. guest room
C. horseradish
D. bathroom
E. living room 


Step 1: They were eating on the sofa bed, which suggests that they were in the living room. Confidence: 90%
Step 2: They dripped a tangy sauce all over it, which implies that the sauce is liquid and easy to spill. Confidence: 80%
Step 3: The most likely place for a liquid spill like this would be on the upholstery or cushions of the sofa bed. Confidence: 95%
Step 4: Therefore, it is likely that they dripped the tangy sauce on the living room sofa bed. Confidence: 90%
Final Answer: C.horseradish
Overall Confidence: 86%
</example>

Question:
{question}
A. {选项A}
B. {选项B}
C. {选项C}
D. {选项D}
E. {选项E} 
Now, please answer this question and provide your confidence level. Let’s think it step by step..
""",
    "CSQA_Verb":"""Read the question, provide your answer and your confidence in this answer. Note: The confidence indicates how likely you think your answer is true.

Use the following format to answer: 
"Answer: [ONLY the option letter; not a complete sentence], 
Confidence (0-100):[Y our confidence level, please only include the numerical number in the range of 0-100]%” 

Only the answer and confidence, don’t give me the explanation.

Please refer to the example I have given:
<example>
The sanctions against the school were a punishing blow, and they seemed to what the efforts the school had made to change?
A. ignore
B. enforce
C. authoritarian
D. yell at
E. avoid
Answer: A. ignore
Confidence: 85%

Young Timmy had a weak arm and could never quite land the newspaper all the way up on my what?
A. library
B. jail
C. subway
D. porch
E. floor 
Answer: E. floor 
Confidence: 80%

After a long night out the drunken man lost consciousness, he showed a sign of sickness right before passing out, what was it?
A. dream
B. vomiting
C. panic
D. cancer
E. blurred vision
Answer: B. vomiting
Confidence: 90%

</example>

Question:
{question}
A. {选项A}
B. {选项B}
C. {选项C}
D. {选项D}
E. {选项E} 
Now, please answer this question and provide your confidence level.

"""
}