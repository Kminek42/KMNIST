import time

def show_time(start_timestamp, progres):
    running_time = int(time.time() - start_timestamp)
    remaining_time = int(running_time / progres - running_time)
    print(f"Running time: {running_time // 3600}h {(running_time // 60) % 60}min {running_time % 60}s")
    print(f"Remaining time: {remaining_time // 3600}h {(remaining_time // 60) % 60}min {remaining_time % 60}s")
