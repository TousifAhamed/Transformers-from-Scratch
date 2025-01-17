def count_lines(filename):
    with open(filename, 'r') as file:
        return sum(1 for _ in file)

def check (filename):
    with open('input.txt', 'r') as f:
        text = f.read()
    print(len(text))

if __name__ == "__main__":
    print(count_lines("input.txt"))
    check("input.txt")