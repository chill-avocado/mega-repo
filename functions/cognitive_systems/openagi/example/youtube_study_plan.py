from openagi.agent import Admin
from openagi.worker import Worker
from openagi.actions.files import WriteFileAction, ReadFileAction
from openagi.actions.tools.youtubesearch import YouTubeSearchTool
from openagi.llms.gemini import GeminiModel
from openagi.memory import Memory
from openagi.planner.task_decomposer import TaskPlanner
import os


# Define the llm
os.environ['GOOGLE_API_KEY'] = "<your-api-key>"
os.environ['Gemini_MODEL'] = "gemini-1.5-flash" # Can use any model you want
os.environ['Gemini_TEMP'] = "0.5"
config = GeminiModel.load_from_env_config()
llm = GeminiModel(config=config)

# Define the planner
planner = TaskPlanner(
    human_intervene=True, # This can be set to false if you want, true here so the planner knows what's your current level, interest, etc.
    llm=llm,
    retry_threshold=5
)

# Define the admin
admin = Admin(
    llm = llm,
    planner=planner,
    max_steps=5,
    memory=Memory(),
    # We want a markdown file, so we will set the output format to markdown
    output_format="markdown"
)

# Define the worker agent
youtube_agent = Worker(
    role="YoutubeResearchAgent",
    instructions=""" 
    Strictly follow these steps:
    1. Generate a detailed study plan for the topic the user wants to study.The topics should start from easy and 
    progress in difficulty level. 
    2. For each topic, provide a very brief description and add 1 youtube video link.
    3. write the whole plan in a file, neatly formatted.
    """,
    actions=[YouTubeSearchTool, WriteFileAction, ReadFileAction],
    force_output= True,
    max_iterations=5
)

admin.assign_workers([youtube_agent])

admin.run(
    query= """I want to study optimization techniques.""",
    description="""Create a detailed studyplan for studying optimization techniques with all the major topics listed 
    along with a reference youtube video for each topic"""
)

########################################################################################################################
#Output:
"""
## Optimization Techniques in Machine Learning and Deep Learning Study Plan

This study plan outlines a structured approach to learning optimization techniques in machine learning and deep learning, progressing from fundamental concepts to more advanced topics.

**1. Introduction to Optimization**

* **Description:** This section introduces the core concepts of optimization, including its role in machine learning, different types of optimization problems, and the fundamental principles of optimization algorithms.
* **YouTube Video:** [https://www.youtube.com/watch?v=5u4G23_OohA](https://www.youtube.com/watch?v=5u4G23_OohA)

**2. Gradient Descent**

* **Description:** Gradient descent is a fundamental optimization algorithm used to minimize a function by iteratively moving in the direction of the steepest descent. This section explores the basic concepts of gradient descent, its variants (e.g., batch, stochastic, mini-batch), and its applications in machine learning.
* **YouTube Video:** [https://www.youtube.com/watch?v=sDv4f4s2SB8](https://www.youtube.com/watch?v=sDv4f4s2SB8)

**3. Convex Optimization**

* **Description:** Convex optimization deals with optimizing functions over convex sets. This section introduces the concepts of convexity, convex functions, and convex optimization problems. It also explores various convex optimization algorithms, such as interior-point methods and gradient projection methods.
* **YouTube Video:** [https://www.youtube.com/watch?v=wr-d4-I026o](https://www.youtube.com/watch?v=wr-d4-I026o)

**4. Stochastic Gradient Descent (SGD)**

* **Description:** Stochastic gradient descent is a widely used optimization algorithm in machine learning, particularly for large datasets. This section delves into the core principles of SGD, its advantages and disadvantages, and its various modifications, such as momentum and adaptive learning rate methods.
* **YouTube Video:** [https://www.youtube.com/watch?v=vMh0z_6Q9rQ](https://www.youtube.com/watch?v=vMh0z_6Q9rQ)

**5. Regularization Techniques**

* **Description:** Regularization techniques are used to prevent overfitting in machine learning models. This section explores different types of regularization, including L1, L2, and Elastic Net regularization, and their impact on model performance.
* **YouTube Video:** [https://www.youtube.com/watch?v=Q81RRH7L2o4](https://www.youtube.com/watch?v=Q81RRH7L2o4)

**6. Advanced Optimization Techniques**

* **Description:** This section delves into more advanced optimization techniques, including:
    * **Second-order methods:** Newton's method, Quasi-Newton methods (e.g., BFGS)
    * **Adaptive optimization algorithms:** Adam, RMSprop, AdaGrad
    * **Optimization for Deep Learning:** Techniques specifically designed for training deep neural networks, such as backpropagation, weight initialization, and batch normalization.
* **YouTube Video:** [https://www.youtube.com/watch?v=IHZwWFHWa-w](https://www.youtube.com/watch?v=IHZwWFHWa-w)

**7. Optimization for Specific Machine Learning Tasks**

* **Description:** This section explores optimization techniques tailored for specific machine learning tasks, such as:
    * **Support Vector Machines (SVMs):** Optimization algorithms for finding the optimal hyperplane to separate data points.
    * **Reinforcement Learning:** Optimization algorithms for training agents to learn optimal policies in dynamic environments.
    * **Generative Adversarial Networks (GANs):** Optimization algorithms for training generative models to create realistic data.
* **YouTube Video:** [https://www.youtube.com/watch?v=P7-y-5-1d5I](https://www.youtube.com/watch?v=P7-y-5-1d5I)

**8. Practical Applications of Optimization**

* **Description:** This section explores real-world applications of optimization techniques in various domains, such as:
    * **Image Recognition:** Training deep neural networks for image classification.
    * **Natural Language Processing:** Optimizing language models for tasks like machine translation and text summarization.
    * **Robotics:** Optimizing robot control algorithms for navigation and manipulation.
* **YouTube Video:** [https://www.youtube.com/watch?v=sDv4f4s2SB8](https://www.youtube.com/watch?v=sDv4f4s2SB8)

**9. Optimization Libraries and Tools**

* **Description:** This section introduces popular optimization libraries and tools used in machine learning and deep learning, such as:
    * **SciPy (Python):** A comprehensive library for scientific computing, including optimization algorithms.
    * **TensorFlow (Python):** A powerful library for deep learning, with built-in optimization functions.
    * **PyTorch (Python):** Another popular deep learning library with flexible optimization capabilities.
* **YouTube Video:** [https://www.youtube.com/watch?v=vMh0z_6Q9rQ](https://www.youtube.com/watch?v=vMh0z_6Q9rQ)

**10. Advanced Topics in Optimization**

* **Description:** This section explores advanced topics in optimization, including:
    * **Constrained Optimization:** Optimization problems with constraints on the variables.
    * **Non-convex Optimization:** Optimization problems with non-convex objective functions.
    * **Multi-objective Optimization:** Optimization problems with multiple objectives to be optimized simultaneously.
* **YouTube Video:** [https://www.youtube.com/watch?v=IHZwWFHWa-w](https://www.youtube.com/watch?v=IHZwWFHWa-w)

**Note:** This study plan is a starting point, and you can customize it based on your specific interests and learning goals. Remember to practice implementing optimization algorithms and experiment with different techniques to gain a deeper understanding of their strengths and limitations.
"""

