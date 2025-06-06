import sys 
def aristarkh_simple():
    q = []
    counts_heigh = {}
    sum = 0
    num = 0
    N = int(sys.stdin.readline())
    for _ in range(N):
        event = sys.stdin.readline().strip()
        
        if event.startswith('+'):
            heigh = int(event[2:])
            q.append(heigh)
            counts_heigh[heigh] = counts_heigh.get(heigh, 0) + 1
            sum += heigh
            num += 1
        elif event == '-':
            if num > 0:
                removed_heigh = q.pop(0)
                counts_heigh[removed_heigh] -= 1
                if counts_heigh[removed_heigh] == 0:
                    del counts_heigh[removed_heigh]
                sum -= removed_heigh
                num -= 1
        if num == 0:
            sys.stdout.write("0\n")
        else:
            average = sum / num
            if average == int(average):
                target_heigh = int(average)
                count = counts_heigh.get(target_heigh, 0)
                sys.stdout.write(f"{count}\n")
            else:
                sys.stdout.write("0\n")
aristarkh_simple()