from setuptools import find_packages, setup

setup(
    name = 'Medical Chatbot',
    version= '1.0.0',
    author= 'Dhruvil Prajapati',
    author_email= 'dhruvilprajapati1154b@gmail.com',
    packages= find_packages(),
    install_requires = [
        'flask==2.3.3',
        'python-dotenv==1.0.0',
        'langchain-groq==0.1.1',
        'groq==0.3.0'
    ]
)