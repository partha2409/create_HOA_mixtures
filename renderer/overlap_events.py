
def can_place(start, end, timeline, MAX_OVERLAP=3):
    for t in range(start, end):
        if timeline[t] >= MAX_OVERLAP:
            return False
    return True

def mark_timeline(start, end, timeline):
    timeline[start:end] += 1