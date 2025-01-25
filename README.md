# Snake Game with AI Agent (Deep Q-Learning)

This project is a classic **Snake Game** implemented with both **Human Mode** and **Agent Mode**. In **Human Mode**, you control the snake yourself, while in **Agent Mode**, an AI (trained using **Deep-Q-Learning**) plays the game autonomously.

## Features:
- **Human Mode**: Play the snake game manually by controlling the snake.
- **Agent Mode**: Train and run an AI agent to play the game using Deep Q-Learning.
- **Customizable Speed**: Adjust the snake's movement speed for both modes.
  
## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/kafleSiroj/snake.git
   cd snake
Set Up a Virtual Environment (Recommended for clean dependencies):

Using venv:

bash
Copy
Edit
python -m venv snake
source snake/bin/activate  # Windows: `snake\Scripts\activate`
Using conda:

bash
Copy
Edit
conda create --n snake python=3.x  # Python 3.7 and above recommended
conda activate snake
Install Dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Usage
Running the Game:
Human Mode: To play the game manually, run:

bash
Copy
Edit
python run.py mode=human speed=x  # Replace x with the desired speed (e.g., 10, 20)
Agent Mode: To run the AI agent, train or test it with:

bash
Copy
Edit
python run.py mode=agent speed=x  # Replace x with the desired speed (e.g., 10, 20)
Enjoy playing and experimenting with both the human-controlled and AI-controlled versions of the Snake game!

vbnet
Copy
Edit

This structure adds a bit more clarity to the functionality of both modes while keeping the se