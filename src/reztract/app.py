import streamlit as st
from langchain_community.chat_models.azureml_endpoint import (
    AzureMLChatOnlineEndpoint,
    AzureMLEndpointApiType,
    CustomOpenAIChatContentFormatter,
)

with st.sidebar:
    st.write("## Prerequisites")
    
    st.write("Before we start, there are certain steps we need to take to deploy the models:")
    
    st.write("- Register for a valid Azure account with subscription\n- Make sure you have access to [Azure AI Studio](https://learn.microsoft.com/en-us/azure/ai-studio/what-is-ai-studio?tabs=home)\n- Create a project and resource group\n- Select Meta-Llama-3 models from Model catalog. This example assumes you are deploying `Meta-Llama-3-70B-Instruct`.\n > Notice that some models may not be available in all the regions in Azure AI and Azure Machine Learning. On those cases, you can create a workspace or project in the region where the models are available and then consume it with a connection from a different one. To learn more about using connections see [Consume models with connections](https://learn.microsoft.com/en-us/azure/ai-studio/how-to/deployments-connections)\n")
    
    st.write("* Deploy with \"Pay-as-you-go\"\nOnce deployed successfully, you should be assigned for an API endpoint and a security key for inference.\nFor more information, you should consult Azure's official documentation [here](https://aka.ms/meta-llama-3-azure-ai-studio-docs) for model deployment and inference.\n")

    st.write("## Instructions")

    st.write("Once you have deployed the model, you can use the following credentials to start using the model (else you should be seeing errors currently):")

    llama3_api_key = st.text_input("Azure Llama 3 API Key", key="llama3_api_key", type="password")
    llama3_endpoint_url = st.text_input("Azure Llama 3 Endpoint URL", key="llama3_endpoint_url")

chat_model = AzureMLChatOnlineEndpoint(
    endpoint_url=llama3_endpoint_url,
    endpoint_api_type=AzureMLEndpointApiType.serverless,
    endpoint_api_key=llama3_api_key,
    content_formatter=CustomOpenAIChatContentFormatter(),
    timeout= 360,
)

from langchain_core.prompts import PromptTemplate
prompt = PromptTemplate(input_variables=["input"], template="""Please extract the following information from the given text and format it as a JSON object according to the ResumeSchema schema. 
                        The text is a resume of an individual. 
                        The schema includes the following fields: name, email, phone, location, skills, education, experience, languages, interests, summary, and achievements.
                        The education, experience, and languages fields are nested fields.
                        The education field has the following subfields: degree, location, details, start_date, end_date.
                        The experience field has the following subfields: title, location, details, start_date, end_date.
                        The languages field has the following subfields: language, proficiency.
                        Following are the types of the fields:
                        - name: str
                        - email: str
                        - phone: str
                        - location: str
                        - skills: list[str]
                        - education: list[Education]
                        - experience: list[Experience]
                        - languages: list[Language]
                        - interests: list[str]
                        - summary: str
                        - achievements: list[str]
                        Please make reasonable assumptions for any missing information. {input}""")

runnable = prompt | chat_model
    

st.title("📝 Resume Extraction and Q&A with Llama 3")

uploaded_file = st.file_uploader("Upload a resume", type=("pdf"))

if uploaded_file and not llama3_api_key:
    st.info("Please add your Llama 3 API key to continue.")

if uploaded_file and llama3_api_key:
    from langchain_community.document_loaders import UnstructuredPDFLoader
    import tempfile
    import shutil

    # Create a temporary file to save the uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmpfile:
        tmpfile.write(uploaded_file.getvalue())
        tmpfile.flush()  # Ensure all data is written to disk
        pdf_path = tmpfile.name  # This is the path to the temporary file

    try:
        loader = UnstructuredPDFLoader(pdf_path)
        docs = loader.load()
    finally:
        # Clean up the temporary file after processing
        shutil.rmtree(pdf_path, ignore_errors=True)
    
    from pydantic import BaseModel

    class Education(BaseModel):
        degree: str
        location: str
        details: str
        start_date: str
        end_date: str

    class Experience(BaseModel):
        title: str
        location: str
        details: str
        start_date: str
        end_date: str

    class Language(BaseModel):
        language: str
        proficiency: str

    class ResumeSchema(BaseModel):
        name: str
        email: str
        phone: str
        location: str
        skills: list[str]
        education: list[Education]
        experience: list[Experience]
        languages: list[Language]
        interests: list[str]
        summary: str
        achievements: list[str]

    from kor import create_extraction_chain, from_pydantic

    schema, validator = from_pydantic(ResumeSchema)

    chain = create_extraction_chain(chat_model, schema, encoder_or_encoder_class="json", validator=validator)

    intermediate = runnable.invoke({"input": docs[0].page_content}).content
    text = chain.invoke({"text": intermediate})["text"]["raw"].replace("<json>", "").replace("</json>", "")

    st.write("### Extracted Information")
    st.write(text)

    question = st.text_input(
        "Ask something about the resume",
        placeholder="What kind of jobs can you recommend me?",
        disabled=not uploaded_file,
    )

    from langchain.chains import LLMChain
    from langchain.memory import ConversationBufferMemory
    from langchain.prompts import (
        ChatPromptTemplate,
        HumanMessagePromptTemplate,
        MessagesPlaceholder,
    )
    from langchain.schema import SystemMessage

    rez_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=f"""You are a professional resume coach having a conversation with a human trying to improve their resume. The following is the information extracted from the resume: {text} 
                Please provide a detailed and personalized feedback based on the user's query:
                """
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{human_input}"),
        ]
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chat_llm_chain = LLMChain(
        llm=chat_model,
        prompt=rez_prompt,
        memory=memory,
        verbose=True,
    )

    if question:
        answer = chat_llm_chain.invoke({"human_input": question})

        st.write("### Answer")
        st.write(answer)


