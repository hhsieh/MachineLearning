from __future__ import print_function
from flask import Flask,render_template, request, redirect
import pandas as pd
import Quandl as Quandl
from bokeh.plotting import figure, show, output_notebook
from bokeh.io import output_notebook
import matplotlib.pyplot as plt


import flask

from bokeh.embed import components
from bokeh.plotting import figure
from bokeh.resources import INLINE
from bokeh.util.string import encode_utf8



app_stock = Flask(__name__)
app_stock.vars={}


@app_stock.route('/index_stock', methods = ['GET','POST'])
def index_stock():
	if request.method == 'GET':
		return render_template('stockquery.html')
	else:
		app_stock.vars['name'] = request.form['name_stock']
		return redirect('/graph_stock')

@app_stock.route('/graph_stock', methods = ['GET'])
def second_stock():	
	n = app_stock.vars['name']
	ss = "WIKI/" + n + ".4"
	mydata = Quandl.get(ss, encoding='latin1', parse_dates=['Date'], dayfirst=True, index_col='Date', trim_start="2016-04-05", trim_end="2016-05-25", returns = "numpy", authtoken="ZemsPswo-xM16GFxuKP2")
	mydata = pd.DataFrame(mydata)
	#mydata['Date'] = mydata['Date'].astype('datetime64[ns]')
	x = mydata['Date']
	y = mydata['Close']
	p = figure(title="Stock close price", x_axis_label='Date', y_axis_label='close price', plot_height = 300, plot_width = 550)
	p.line(x, y, legend="Price in USD", line_width=3, color = "#2222aa")
	
	
	# Configure resources to include BokehJS inline in the document.
    # For more details see:
    #   http://bokeh.pydata.org/en/latest/docs/reference/resources_embedding.html#bokeh-embed
	js_resources = INLINE.render_js()
	css_resources = INLINE.render_css()

    # For more details see:
    #   http://bokeh.pydata.org/en/latest/docs/user_guide/embedding.html#components
	script, div = components(p, INLINE)
    
	html = flask.render_template(
		'stockgraph.html',
		ticker = app_stock.vars['name'],
		plot_script=script,
		plot_div=div,
		js_resources=js_resources,
		css_resources=css_resources,
	)
	return encode_utf8(html)
	
		

if __name__ == "__main__":
    app_stock.run(debug=True)