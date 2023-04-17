import random
import numpy as np
from scipy.special import softmax
import os
from itertools import product
# Constants
POPULATION_SIZE = 500
MUTATION_RATE = 0.01
GENERATIONS = 100
MIN_IMPROVEMENT = 0.01


class Activity:
    def __init__(self, name, expected_enrollment, preferred_facilitators, other_facilitators, time_slot=None):
        self.name = name
        self.expected_enrollment = expected_enrollment
        self.preferred_facilitators = preferred_facilitators
        self.other_facilitators = other_facilitators
        self.time_slot = time_slot

    def calculate_fitness(self, room, facilitator, conflicts, facilitator_count, consecutive_facilitator, time_slots):
        fitness = 0

        # Check for room and facilitator conflicts
        for conflict_activity_pair, conflict_type in conflicts.items():
            if self in conflict_activity_pair:
                if conflict_type == "room_conflict":
                    fitness -= 1.0  # Increase the penalty for room conflicts
                elif conflict_type == "facilitator_conflict":
                    fitness -= 0.5  # Increase the penalty for facilitator conflicts

        # Check for room size
        if room.capacity < self.expected_enrollment:
            fitness -= 0.5
        elif room.capacity > 3 * self.expected_enrollment:
            fitness -= 0.2
        elif room.capacity > 6 * self.expected_enrollment:
            fitness -= 0.4
        else:
            fitness += 0.5  # Increase the reward for optimal room size

        # Check for facilitator preferences
        if facilitator in self.preferred_facilitators:
            fitness += 1.0  # Increase the reward for preferred facilitators
        elif facilitator in self.other_facilitators:
            fitness += 0.2
        else:
            fitness -= 0.1

        # Check for facilitator load
        if facilitator_count[facilitator] == 1:
            fitness += 0.2
        elif facilitator_count[facilitator] > 4:
            fitness -= 0.5
        elif facilitator_count[facilitator] in [1, 2] and facilitator.id != "Tyler":
            fitness -= 0.4

        if consecutive_facilitator[facilitator]:
            if self.name.startswith("SLA191") or self.name.startswith("SLA101"):
                fitness += 0.5
                if room.id.startswith("Roman") or room.id.startswith("Beach"):
                    fitness -= 0.4

        # Activity-specific adjustments
        if self.name.startswith("SLA101") or self.name.startswith("SLA191"):
            sections = [a for a in activities if a.name.startswith(self.name[:7])]
            for section in sections:
                if section != self:
                    section_time_slot_index = time_slots.index(section.time_slot)
                    current_time_slot_index = time_slots.index(self.time_slot)
                    diff = abs(section_time_slot_index - current_time_slot_index)

                    if diff >= 4:
                        fitness += 0.5
                    elif diff == 0:
                        fitness -= 0.5
                    elif diff == 1:
                        fitness += 0.5
                        if room.id.startswith("Roman") or room.id.startswith("Beach"):
                            fitness -= 0.4
                    elif diff == 2:
                        fitness += 0.25
                    elif diff == 3:
                        fitness -= 0.25

        return fitness

class Room:
    def __init__(self, id, capacity):
        self.id = id
        self.capacity = capacity


class Facilitator:
    def __init__(self, id, name):
        self.id = id
        self.name = name



def initialize_population(activities, rooms, facilitators, time_slots):
    population = []

    for _ in range(POPULATION_SIZE):
        individual = []

        for activity in activities:
            assigned_room = random.choice(rooms)
            assigned_facilitator = random.choice(facilitators)
            assigned_time_slot = random.choice(time_slots)

            individual.append((activity, assigned_room, assigned_facilitator, assigned_time_slot))

        population.append(individual)

    return population


def calculate_fitness(individual, activities, rooms, facilitators, time_slots):
    fitness = 0
    conflicts = {}
    facilitator_count = {f: 0 for f in facilitators}
    consecutive_facilitator = {f: False for f in facilitators}

    for i, (activity, room, facilitator, time_slot) in enumerate(individual):
        activity.time_slot = time_slot
        facilitator_count[facilitator] += 1

        for j, (other_activity, other_room, other_facilitator, other_time_slot) in enumerate(individual):
            if i != j and time_slot == other_time_slot:
                if room == other_room:
                    conflicts[frozenset({activity, other_activity})] = "room_conflict"
                if facilitator == other_facilitator:
                    conflicts[frozenset({activity, other_activity})] = "facilitator_conflict"

        if i > 0:
            prev_activity, prev_room, prev_facilitator, prev_time_slot = individual[i - 1]
            if facilitator == prev_facilitator and abs(time_slots.index(time_slot) - time_slots.index(prev_time_slot)) == 1:
                consecutive_facilitator[facilitator] = True

        fitness += activity.calculate_fitness(room, facilitator, conflicts, facilitator_count, consecutive_facilitator, time_slots)

    return fitness


def selection(population, activities, rooms, facilitators, time_slots):
    fitness_scores = [calculate_fitness(individual, activities, rooms, facilitators, time_slots) for individual in population]
    probabilities = softmax(fitness_scores)
    selected_indices = np.random.choice(len(population), size=2, p=probabilities)
    return population[selected_indices[0]], population[selected_indices[1]]


def crossover(parent1, parent2, activities):
    crossover_point = random.randint(1, len(activities) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2


def mutate(individual, activities, rooms, facilitators, time_slots):
    for i, (activity, room, facilitator, time_slot) in enumerate(individual):
        if random.random() < MUTATION_RATE:
            if random.random() < 0.33:
                individual[i] = (activity, random.choice(rooms), facilitator, time_slot)
            elif random.random() < 0.66:
                individual[i] = (activity, room, random.choice(facilitators), time_slot)
            else:
                individual[i] = (activity, room, facilitator, random.choice(time_slots))
    return individual


activities = [
        Activity("SLA100A", 50, ["Glen", "Lock", "Banks", "Zeldin"], ["Numen", "Richards"]),
        Activity("SLA100B", 50, ["Glen", "Lock", "Banks", "Zeldin"], ["Numen", "Richards"]),
        Activity("SLA191A", 50, ["Glen", "Lock", "Banks", "Zeldin"], ["Numen", "Richards"]),
        Activity("SLA191B", 50, ["Glen", "Lock", "Banks", "Zeldin"], ["Numen", "Richards"]),
        Activity("SLA201", 50, ["Glen", "Banks", "Zeldin", "Shaw"], ["Numen", "Richards", "Singer"]),
        Activity("SLA291", 50, ["Lock", "Banks", "Zeldin", "Singer"], ["Numen", "Richards", "Shaw", "Tyler"]),
        Activity("SLA303", 60, ["Glen", "Zeldin", "Banks"], ["Numen", "Singer", "Shaw"]),
        Activity("SLA304", 25, ["Glen", "Banks", "Tyler"], ["Numen", "Singer", "Shaw", "Richards", "Uther", "Zeldin"]),
        Activity("SLA394", 20, ["Tyler", "Singer"], ["Richards", "Zeldin"]),
        Activity("SLA449", 60, ["Tyler", "Singer", "Shaw"], ["Zeldin", "Uther"]),
        Activity("SLA451", 100, ["Tyler", "Singer", "Shaw"], ["Zeldin", "Uther", "Richards", "Banks"]),
    ]

rooms = [
        Room("Slater 003", 45),
        Room("Roman 216", 30),
        Room("Loft 206", 75),
        Room("Roman 201", 50),
        Room("Loft 310", 108),
        Room("Beach 201", 60),
        Room("Beach 301", 75),
    ]

facilitators = [
    Facilitator("Lock", "Lock"),
    Facilitator("Glen", "Glen"),
    Facilitator("Banks", "Banks"),
    Facilitator("Richards", "Richards"),
    Facilitator("Shaw", "Shaw"),
    Facilitator("Singer", "Singer"),
    Facilitator("Uther", "Uther"),
    Facilitator("Tyler", "Tyler"),
    Facilitator("Numen", "Numen"),
    Facilitator("Zeldin", "Zeldin"),
]


time_slots = [
        "10 AM",
        "11 AM",
        "12 PM",
        "1 PM",
        "2 PM",
        "3 PM",
    ]


def save_schedule_to_txt(best_individual, filename="schedule.txt"):
    with open(filename, "w") as file:
        for activity, room, facilitator, time_slot in best_individual:
            file.write(f"{activity.name}, {room.id}, {facilitator.name}, {time_slot}\n")

def main():
    best_overall_fitness = -float('inf')
    best_overall_individual = None

    num_runs = 10

    for run in range(num_runs):
        print(f"Run {run + 1}:")

        # Run the genetic algorithm
        population = initialize_population(activities, rooms, facilitators, time_slots)

        for generation in range(GENERATIONS):
            new_population = []

            while len(new_population) < POPULATION_SIZE:
                parent1, parent2 = selection(population, activities, rooms, facilitators, time_slots)
                child1, child2 = crossover(parent1, parent2, activities)

                child1 = mutate(child1, activities, rooms, facilitators, time_slots)
                child2 = mutate(child2, activities, rooms, facilitators, time_slots)

                new_population.append(child1)
                new_population.append(child2)

            population = new_population

            # Check for convergence
            fitness_scores = [calculate_fitness(individual, activities, rooms, facilitators, time_slots) for individual in population]
            best_individual = population[np.argmax(fitness_scores)]
            best_fitness = max(fitness_scores)

            if generation > 0 and (best_fitness - prev_best_fitness) < MIN_IMPROVEMENT:
                break

            prev_best_fitness = best_fitness

        # Output the best individual of this run
        print("Best fitness:", best_fitness)

        # Save the best individual from all runs
        if best_fitness > best_overall_fitness:
            best_overall_fitness = best_fitness
            best_overall_individual = best_individual

    # Output the best individual from all runs
    print("\nBest overall fitness:", best_overall_fitness)

    # Save the best overall schedule to a .txt file
    save_schedule_to_txt(best_overall_individual)

if __name__ == "__main__":
    main()




