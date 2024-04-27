from getpass import getpass
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
import os

question = "What is the best restaurant in Dallas?"

template = """Question: {question} """ #Add additional context or words to template

prompt = PromptTemplate.from_template(template)

repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

llm = HuggingFaceEndpoint(
    repo_id=repo_id, max_length=128, temperature=0.5, huggingfacehub_api_token="YOUR TOKEN"
)
llm_chain = LLMChain(prompt=prompt, llm=llm)
print(llm_chain.run(question))


# The "best" restaurant can depend on various factors such as personal taste, budget, and specific dietary needs. Here are a few highly-regarded restaurants in Dallas that cater to different cuisines and dining experiences:

# 1. Lucia: This Italian restaurant, located in the Bishop Arts District, is known for its handmade pastas and seasonal dishes. It's a great choice for those who enjoy authentic Italian cuisine.

# 2. Mirador: Situated on the 27th floor of the Magnolia Hotel, Mirador offers a breathtaking view of the city along with innovative American cuisine. It's a perfect place for a special occasion or a romantic dinner.

# 3. Tei-An: For Japanese food lovers, Tei-An is a must-visit. This authentic Japanese tea house and restaurant in the Arts District serves delicious and beautifully presented dishes.

# 4. Spice: Spice, located in the Design District, offers contemporary Indian cuisine. The restaurant is known for its bold flavors and creative presentation.

# 5. Javier's Gourmet Mexicano: If you're looking for Mexican food, Javier's in Uptown Dallas is a popular choice. The restaurant is famous for its signature dish, the "Filet Mignon a la Tampiquena," and its extensive tequila selection.

# Ultimately, the best restaurant in Dallas depends on your personal preferences and dining needs. It's always a good idea to check out reviews and make a reservation in advance for a memorable dining experience.