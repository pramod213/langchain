import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint # pyright: ignore[reportMissingImports]
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id = "google/gemma-2-2b-it",
    task = "text-generation"
)

model1 = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",  
    api_key=os.getenv("GOOGLE_API_KEY")
)

model2 = ChatHuggingFace(llm = llm)

prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate a  5 short question answers from the following text \n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes' , 'quiz']
)

parser = StrOutputParser()

# Creating two parallel chains
parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser ,
    'quiz': prompt2 | model2 | parser 
})

#Merging the chains

merge_chain = prompt3 | model1 | parser 

# Creating the final chain by joining the parallel and merging chain 

chain = parallel_chain | merge_chain

text = '''What is Support Vector Machine (SVM)
Support vector machines (SVMs) are powerful yet flexible supervised machine learning algorithm which is used for both classification and regression. But generally, they are used in classification problems. In 1960s, SVMs were first introduced but later they got refined in 1990 also. SVMs have their unique way of implementation as compared to other machine learning algorithms. Now a days, they are extremely popular because of their ability to handle multiple continuous and categorical variables.

Working of SVM
The goal of SVM is to find a hyperplane that separates the data points into different classes. A hyperplane is a line in 2D space, a plane in 3D space, or a higher-dimensional surface in n-dimensional space. The hyperplane is chosen in such a way that it maximizes the margin, which is the distance between the hyperplane and the closest data points of each class. The closest data points are called the support vectors.

The distance between the hyperplane and a data point "x" can be calculated using the formula −

distance = (w . x + b) / ||w|| 
where "w" is the weight vector, "b" is the bias term, and "||w||" is the Euclidean norm of the weight vector. The weight vector "w" is perpendicular to the hyperplane and determines its orientation, while the bias term "b" determines its position.

The optimal hyperplane is found by solving an optimization problem, which is to maximize the margin subject to the constraint that all data points are correctly classified. In other words, we want to find the hyperplane that maximizes the margin between the two classes while ensuring that no data point is misclassified. This is a convex optimization problem that can be solved using quadratic programming.

If the data points are not linearly separable, we can use a technique called kernel trick to map the data points into a higher-dimensional space where they become separable. The kernel function computes the inner product between the mapped data points without computing the mapping itself. This allows us to work with the data points in the higherdimensional space without incurring the computational cost of mapping them.

'''

result = chain.invoke({'text': text})

print(result)

#Vizualizing the chain 
chain.get_graph().print_ascii()