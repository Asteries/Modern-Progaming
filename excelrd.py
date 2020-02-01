import xlrd


def excelread():
    data = []
    sum_sheet = []
    tot = []
    xls_str = list("C:\\Users\\Asteries\\PycharmProjects\\OO\\co2_demo\\Province sectoral CO2 emissions 1997.xlsx")
    for i in range(19):
        data.append(xlrd.open_workbook("".join(xls_str)))
        xls_str[-9:-5] = list(str(int("".join(xls_str[-9:-5])) + 1))
        sum_sheet.append(data[i].sheet_by_name("Sum"))
        tot.append(sum_sheet[i].col_values(1, start_rowx=1, end_rowx=31))
    return [sum_sheet, tot]
