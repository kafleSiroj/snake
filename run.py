from agent.agent import Agent
from human_mode.human import Game
from environment.env import GameAgent
import sys

def launchAgent(speed):
    best = 0
    agent = Agent()
    environment = GameAgent(speed=speed)

    while True:
        state = agent.state(environment)

        action = agent.action(state)

        done, score, reward = environment.play(action)
        next_state = agent.state(environment)

        #short memory
        agent.train_short(state, action, reward, next_state, done)

        agent.experience(state, action, reward, next_state, done)
        
        if done:
            environment.reset_func()
            agent.n_games += 1
            agent.train_long()

            if score > best:
                best = score
            
            print(f"Game: {agent.n_games} | Score: {score}| Best: {best}")


def launchHuman(speed):
    game = Game(speed=speed)
    best = 0

    while True:
        game_over, score = game.play()

        if game_over:
            if score > best:
                best = score

            print(f"Score: {score} | Best: {best}")
            game.replay()

def run(argv):
    if len(argv) == 1:
        launchHuman(20)

    for i in range(len(argv)):
        if argv[i] in (' ', ''):
            argv.pop(i)

    if len(argv)>3:
        print("Cannot take more than 2 argument! Please don't use spaces before or after '='")
        return

    if len(argv)>1 and len(argv)<3:
        print("['mode=' and 'speed='] is required!")
        return

    modes = ['mode=human', 'mode=agent']

    args = sorted(argv[1:])


    if args[0] not in modes:
        print("Invalid syntax. Please use: ['mode=human' or 'mode=agent']")
        return

    speedArgs = args[1].split('=')

    if not speedArgs[-1].isdigit():
        print("Value Error. Speed must be a integer.")
        return

    speed = int(speedArgs[-1])

    if args[0] == modes[0]:
        launchHuman(speed=speed)

    elif args[0] == modes[1]:
        launchAgent(speed=speed)


if __name__ == '__main__':
    run(sys.argv)