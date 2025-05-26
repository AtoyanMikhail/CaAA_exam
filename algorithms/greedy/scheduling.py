from typing import List, Tuple
from dataclasses import dataclass
from operator import attrgetter

@dataclass
class Job:
    """Задача с временем выполнения и дедлайном"""
    id: int
    duration: int  # время выполнения
    deadline: int  # дедлайн
    weight: float  # вес/приоритет задачи

def schedule_jobs(jobs: List[Job]) -> List[Tuple[int, int]]:
    """
    Реализация жадного алгоритма для задачи об оптимальном расписании.
    Сортирует задачи по убыванию веса/времени выполнения.
    
    Args:
        jobs: Список задач
        
    Returns:
        Список пар (id_задачи, время_начала) - оптимальное расписание
    """
    # Сортируем задачи по убыванию веса/времени выполнения
    sorted_jobs = sorted(jobs, key=lambda x: x.weight/x.duration, reverse=True)
    
    current_time = 0
    schedule = []
    
    for job in sorted_jobs:
        schedule.append((job.id, current_time))
        current_time += job.duration
        
    return schedule

def schedule_with_deadlines(jobs: List[Job]) -> List[Tuple[int, int]]:
    """
    Реализация алгоритма для задачи с дедлайнами.
    Сортирует задачи по дедлайнам и пытается выполнить их как можно раньше.
    
    Args:
        jobs: Список задач
        
    Returns:
        Список пар (id_задачи, время_начала) - оптимальное расписание
    """
    # Сортируем задачи по дедлайнам
    sorted_jobs = sorted(jobs, key=attrgetter('deadline'))
    
    current_time = 0
    schedule = []
    
    for job in sorted_jobs:
        # Если задача не может быть выполнена до дедлайна, пропускаем её
        if current_time + job.duration <= job.deadline:
            schedule.append((job.id, current_time))
            current_time += job.duration
            
    return schedule

def schedule_with_weights(jobs: List[Job]) -> List[Tuple[int, int]]:
    """
    Реализация алгоритма для задачи с весами.
    Сортирует задачи по убыванию веса/времени выполнения.
    
    Args:
        jobs: Список задач
        
    Returns:
        Список пар (id_задачи, время_начала) - оптимальное расписание
    """
    # Сортируем задачи по убыванию веса/времени выполнения
    sorted_jobs = sorted(jobs, key=lambda x: x.weight/x.duration, reverse=True)
    
    current_time = 0
    schedule = []
    
    for job in sorted_jobs:
        schedule.append((job.id, current_time))
        current_time += job.duration
        
    return schedule

if __name__ == "__main__":
    # Пример использования
    jobs = [
        Job(1, 3, 5, 2.0),  # id=1, duration=3, deadline=5, weight=2.0
        Job(2, 2, 3, 1.5),  # id=2, duration=2, deadline=3, weight=1.5
        Job(3, 4, 6, 3.0),  # id=3, duration=4, deadline=6, weight=3.0
        Job(4, 1, 2, 1.0),  # id=4, duration=1, deadline=2, weight=1.0
    ]
    
    print("Расписание по весам/времени:")
    schedule = schedule_with_weights(jobs)
    for job_id, start_time in schedule:
        print(f"Задача {job_id}: начало в {start_time}")
    
    print("\nРасписание с учетом дедлайнов:")
    schedule = schedule_with_deadlines(jobs)
    for job_id, start_time in schedule:
        print(f"Задача {job_id}: начало в {start_time}") 