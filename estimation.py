import tensorflow as tf
import numpy as np
import math
from random import sample
import random
import math
from sklearn import preprocessing

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Lambda, Concatenate, Dense

from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.constraints import non_neg

np.random.seed(42)
random.seed(42)
scale = 400 / math.log(10)
number_of_matches = 10000
team_size = 3
mu, sigma, number_of_people = 1200, 400, 100

real_skills = np.random.normal(mu, sigma, number_of_people)


def generate_data(real_skills):
    def team_skill_generalized_power_mean(gen_mean_m):
        def _mean(skills):
            return (sum(map(lambda x: x ** gen_mean_m, skills)) / len(skills)) ** (1 / gen_mean_m)

        return _mean

    power_mean = team_skill_generalized_power_mean(1)

    matches = []

    for i in range(number_of_matches):
        teamA = sample(range(number_of_people), team_size)
        teamB = sample(range(number_of_people), team_size)

        skillsA = [real_skills[i] for i in teamA]
        skillsB = [real_skills[i] for i in teamB]

        skillA = power_mean(skillsA)
        skillB = power_mean(skillsB)

        A_score = np.random.logistic(skillA, scale / 2, 1)
        B_score = np.random.logistic(skillB, scale / 2, 1)
        result = int(A_score > B_score)

        matches.append((teamA, teamB, result))

    teamsA = [match[0] for match in matches]
    teamsB = [match[1] for match in matches]
    results = [match[2] for match in matches]

    teamsA = np.array(teamsA)
    teamsB = np.array(teamsB)
    results = np.array(results)[:, None]

    data = np.concatenate([teamsA, teamsB, results], axis=1)
    return data


data = generate_data(real_skills)


def create_model():
    input_layer = Input(team_size * 2)
    embedding = Embedding(100, 1)(input_layer)
    embedding = Flatten()(embedding)
    team_a_embedding = embedding[:, :team_size]
    team_b_embedding = embedding[:, team_size:]

    dense = Dense(5, kernel_constraint=non_neg())
    dense2 = Dense(5, kernel_constraint=non_neg())
    densefinal = Dense(1, kernel_constraint=non_neg())

    def multidense(x):
        return densefinal(dense2(dense(x)))

    # multidense =  Lambda(lambda x: tf.math.reduce_sum(x,axis = 1, keepdims = True)/3)

    team_a_elo = multidense(team_a_embedding)
    team_b_elo = multidense(team_b_embedding)
    elos = Concatenate(axis=1)([team_a_elo, team_b_elo])

    def loglike(win, elos):
        win = tf.cast(win, tf.float32)
        teamAelo = elos[:, 0]
        teamBelo = elos[:, 1]
        win = tf.reshape(win, [-1])
        teamAlikelihood = 1 - 1 / (1 + math.e ** ((teamAelo - teamBelo) / scale))
        teamBlikelihood = 1 / (1 + math.e ** ((teamAelo - teamBelo) / scale))
        loglike = win * tf.math.log(teamAlikelihood) + (1 - win) * tf.math.log(teamBlikelihood)
        # import pdb; pdb.set_trace()
        return - tf.math.reduce_sum(loglike)

    opt = Adam(0.1)
    model = Model(input_layer, elos)
    model.compile(optimizer=opt, loss=loglike)
    return model


model = create_model()
model.fit(data[:, :team_size * 2], data[:, team_size * 2], epochs=50, batch_size=300)


def calculate_metrics(estimated_skills, real_skills):
    def ordering(estimated_skill, real_skills):
        estimated_skill = np.array(estimated_skill)
        real_skills = np.array(real_skills)
        estimated_skill_order = estimated_skill > estimated_skill[:, None]
        real_skills_order = real_skills > real_skills[:, None]
        return np.sum(np.invert(np.logical_xor(estimated_skill_order, real_skills_order))) / len(real_skills_order) ** 2

    std = scale * math.pi / math.sqrt(3)
    scaled = preprocessing.scale(model.get_weights()[0].reshape(-1, 1))[:, 0]
    estimated_real = scaled * std + mu

    print(ordering(estimated_skills, real_skills))
    print(list(zip(estimated_real, real_skills)))


estimated_skills = model.get_weights()[0].reshape(-1)
calculate_metrics(estimated_skills, real_skills)
