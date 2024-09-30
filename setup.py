from setuptools import find_packages, setup

setup(
    name='Medical-Chatbot',
    version='0.0.1',
    author='Hector S.',
    author_email='',
    install_requires = ["langchain==0.3.1",
                        "langchain_community==0.3.1",
                        "langchain-groq==0.2.0",
                        "python-dotenv==1.0.1", 
                        "pypdf==5.0.1",
                        "pinecone-client==0.5.1",
                        "flask"],
    packages=find_packages()
)