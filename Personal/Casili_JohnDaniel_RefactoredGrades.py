def get_remark(average):
    if average >= 90:
        return "Excellent"
    elif average >= 80:
        return "Very Good"
    elif average >= 75:
        return "Good"
    elif average >= 60:
        return "Needs Improvement"
    else:
        return "Poor"


def get_status(average):
    return "Passed" if average >= 75 else "Failed"


def get_quiz_score(prompt):
    raw = input(prompt).strip()
    return float(raw) if raw else 0.0


def compute_average(*scores):
    return sum(scores) / len(scores)


def input_student():
    name = input("Enter student name or type STOP to finish: ").strip()
    if name.lower() == "stop":
        return None

    score1 = get_quiz_score("  Enter Quiz 1 score: ")
    score2 = get_quiz_score("  Enter Quiz 2 score: ")
    score3 = get_quiz_score("  Enter Quiz 3 score: ")

    average = compute_average(score1, score2, score3)
    remark  = get_remark(average)
    status  = get_status(average)

    return {
        "name":    name,
        "average": average,
        "remark":  remark,
        "status":  status,
    }


def print_student_result(student):
    print()
    print(f"  Student Name : {student['name']}")
    print(f"  Average      : {student['average']:.2f}")
    print(f"  Remark       : {student['remark']}")
    print(f"  Status       : {student['status']}")
    print()


def get_class_performance(passing_rate):
    if passing_rate >= 80:
        return "Strong"
    elif passing_rate >= 50:
        return "Moderate"
    else:
        return "Needs Attention"


def print_summary(records):
    total  = len(records)
    passed = sum(1 for r in records if r["status"] == "Passed")
    failed = total - passed
    passing_rate = (passed / total * 100) if total > 0 else 0

    print("\nSUMMARY REPORT")
    print("-" * 40)
    for student in records:
        print(f"Name    : {student['name']}")
        print(f"Average : {student['average']:.2f}")
        print(f"Remark  : {student['remark']}")
        print(f"Status  : {student['status']}")
        print("-" * 40)

    print(f"Total Students    : {total}")
    print(f"Passed            : {passed}")
    print(f"Failed            : {failed}")
    print(f"Passing Rate      : {passing_rate:.1f} %")
    print(f"Class Performance : {get_class_performance(passing_rate)}")


def main():
    print("STUDENT GRADE PROCESSING SYSTEM")
    print("-" * 40)

    records = []

    while True:
        student = input_student()
        if student is None:
            break
        records.append(student)
        print_student_result(student)

    print_summary(records)


if __name__ == "__main__":
    main()