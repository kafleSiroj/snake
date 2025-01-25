# Snake
This is a simple snake game, we can play in **Human Mode** as well as train an **Agent**. Used Deep-Q-Learning to train the Agent

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/kafleSiroj/snake.git
   cd snake
   ```

2. Create a virtual environment (recommended):
    ### Using venv
   ```bash
   python -m venv snake
   source snake/bin/activate  # Windows: `snake\Scripts\activate`
   ```
   ### Using conda
   ```bash
   conda create --n snake python=3.x # Python 3.7 and above recommended
   conda activate snake
   ```


3. Install required dependencies:

   ```bash
   pip install -r requirements.txt
   ```
   
## Usage

1. Run `run.py`
   ### Human Mode
   ```bash
   python run.py  # Default mode without any arguments is human mode with speed 20`
   ```
   or
   ```bash
   python run.py mode=human speed=x # Speed should have an integer value i.e. `10`, `20`
   ```
   ### Agent Mode
   ```bash
   python run.py mode=agent speed=x # Speed should have an integer value i.e. `10`, `20`
   ```