from pyecharts.charts import Bar
from pyecharts import options as opts


def draw_bar(sum_sheet):
    prov = sum_sheet[0].col_values(0, start_rowx=1, end_rowx=31)
    line = Bar()
    x1 = []
    y1 = []
    y2 = []
    for i in range(19):
        x1.append(1997 + i)
        y1.append(sum_sheet[i].cell_value(1, 1))
        y2.append(sum_sheet[i].cell_value(9, 1))
    line.add_xaxis(x1)
    line.set_global_opts(xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(font_size=10, interval=0, rotate=45)))
    line.add_yaxis("Beijing", y1)
    line.add_yaxis("Shanghai", y2)
    line.render("北京上海年CO2排放量变化图.html")

    bar = Bar()
    x2 = []
    for i in range(30):
        x2.append(prov[i])
    y3 = sum_sheet[0].col_values(1, start_rowx=1, end_rowx=31)
    y4 = sum_sheet[-1].col_values(1, start_rowx=1, end_rowx=31)
    bar.add_xaxis(x2)
    bar.set_global_opts(xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(font_size=10, interval=0, rotate=45)))
    bar.add_yaxis("1997", y3)
    bar.add_yaxis("2015", y4)
    bar.render("1997与2015 各省CO2年排放量对比.html")
