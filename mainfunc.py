import excelrd
import makestat
import visual
import json


def main():
    sum_sheet, tot = excelrd.excelread()
    with open('sums.json', 'w') as f:
        json.dump(tot, f)
    makestat.print_stat(sum_sheet)
    visual.draw_bar(sum_sheet)


if __name__ == "__main__":
    main()
