'use strict';


// http://www.d3noob.org/2013/01/adding-tooltips-to-d3js-graph.html


var LDAvis = function(to_select, data_or_file_name) {

	// This section sets up the logic for event handling
	var current_clicked = {
		what: "nothing",
		element: undefined
	},
		current_hover = {
			what: "nothing",
			element: undefined
		},
		old_winning_state = {
			what: "nothing",
			element: undefined
		},
		vis_state = {
			doc: 0,
			topic: 0,
			word: 0
		};

	// Set up a few 'global' variables to hold the data:
	var D, // number of docs
		T, // number of topics
		W, // number of words
		docMdsData, // (x,y) locations and topic proportions
		topicMdsData,
		wordMdsData,
		doc_topic_info, // topic proportions for all docs in the viz
		doc_word_info,
		topic_doc_info,
		topic_word_info,
		word_doc_info,
		word_topic_info,
		color1 = "#1f77b4", // baseline color for default topic circles and overall word frequencies
		color2 = "#d62728"; // 'highlight' color for selected topics and word-topic frequencies

	// Set the duration of each half of the transition:
	var duration = 750;

	// Set global margins used for everything
	var margin = {
		top: 30,
		right: 30,
		bottom: 70,
		left: 5,
		pad: 10
	},
		mdswidth = 390,
		mdsheight = 530,
		mdsarea = mdsheight * mdswidth;
	// controls how big the maximum circle can be
	// doesn't depend on data, only on mds width and height:
	var rMax = 72;

	// proportion of area of MDS plot to which the sum of default topic circle areas is set
	var circle_prop = 0.25;

	// opacity of topic circles:
	var base_opacity = 0.2,
		highlight_opacity = 0.6;

	// doc/topic/word selection names are specific to *this* vis
	var doc_select = to_select + "-doc";
	var topic_select = to_select + "-topic";
	var word_select = to_select + "-word";

	// get rid of the # in the to_select (useful) for setting ID values
	var visID = to_select.replace("#", "");
	var topID = visID + "-top";
	var docID = visID + "-doc";
	var topicID = visID + "-topic";
	var wordID = visID + "-word";
	// ---------
	var docDown = docID + "-down";
	var docUp = docID + "-up";
	var docClear = docID + "-clear";
	var docShow = docID + "-show";
	var topicDown = topicID + "-down";
	var topicUp = topicID + "-up";
	var topicClear = topicID + "-clear";
	var wordDown = wordID + "-down";
	var wordUp = wordID + "-up";
	var wordClear = wordID + "-clear"; 

	var docPanelID = visID + "-docPanel";
	var topicPanelID = visID + "-topicPanel";
	var wordPanelID = visID + "-wordPanel";

	var DocTextID = visID + "-docText";
	var DocTextBoundaryID = visID + "-docTextBoundary";

	//////////////////////////////////////////////////////////////////////////////


	function visualize(data) {

		// set the number of documents to global variable D:
		D = data['doc_mds'].x.length;
		// // set the number of topics to global variable T:
		T = data['topic_mds'].x.length;
		// set the number of words to global variable W:
		W = data['word_mds'].x.length;

		// a (D x 3) matrix with columns x, y, doc_tag
		docMdsData = [];
		for (var i = 0; i < D; i++) {
			var obj = {};
			for (var key in data['doc_mds']) {
				obj[key] = data['doc_mds'][key][i];
			}
			docMdsData.push(obj);
		}

		// a (T x 4) matrix with columns x, y, topics id, Freq
		topicMdsData = [];
		for (var i = 0; i < T; i++) {
			var obj = {};
			for (var key in data['topic_mds']) {
				obj[key] = data['topic_mds'][key][i];
			}
			topicMdsData.push(obj);
		}

		// a (W x 4) matrix with columns x, y, vocab word, Freq
		wordMdsData = [];
		for (var i = 0; i < W; i++) {
			var obj = {};
			for (var key in data['word_mds']) {
				obj[key] = data['word_mds'][key][i];
			}
			wordMdsData.push(obj);
		}


		doc_topic_info = [];
		for (var i = 0; i < data['doc_topic.info'].Doc.length; i++) {
			var obj = {};
			for (var key in data['doc_topic.info']) {
				obj[key] = data['doc_topic.info'][key][i];
			}
			doc_topic_info.push(obj);
		}

		doc_word_info = [];
		for (var i = 0; i < data['doc_word.info'].Doc.length; i++) {
			var obj = {};
			for (var key in data['doc_word.info']) {
				obj[key] = data['doc_word.info'][key][i];
			}
			doc_word_info.push(obj);
		}

		topic_doc_info = [];
		for (var i = 0; i < data['topic_doc.info'].Topic.length; i++) {
			var obj = {};
			for (var key in data['topic_doc.info']) {
				obj[key] = data['topic_doc.info'][key][i];
			}
			topic_doc_info.push(obj);
		}

		topic_word_info = [];
		for (var i = 0; i < data['topic_word.info'].Topic.length; i++) {
			var obj = {};
			for (var key in data['topic_word.info']) {
				obj[key] = data['topic_word.info'][key][i];
			}
			topic_word_info.push(obj);
		}

		word_doc_info = [];
		for (var i = 0; i < data['word_doc.info'].Word.length; i++) {
			var obj = {};
			for (var key in data['word_doc.info']) {
				obj[key] = data['word_doc.info'][key][i];
			}
			word_doc_info.push(obj);
		}

		word_topic_info = [];
		for (var i = 0; i < data['word_topic.info'].Word.length; i++) {
			var obj = {};
			for (var key in data['word_topic.info']) {
				obj[key] = data['word_topic.info'][key][i];
			}
			word_topic_info.push(obj);
		}

		docMdsData.sort(function(a, b) {
		    return a.Freq - b.Freq;
		});
		wordMdsData.sort(function(a, b) {
		    return a.Freq - b.Freq;
		});

		// Create the doc/topic/word input forms
		init_forms(docID, topicID, wordID);


		d3.select("#" + docID)
			.on("keyup", function() {
				// remove topic selection if it exists (from a saved URL)
				var topicElem = document.getElementById(topicID + vis_state.topic);
				if (topicElem !== undefined) topic_off(topicElem);
				vis_state.topic = 0;
				// remove word selection if it exists (from a saved URL)
				var wordElem = document.getElementById(wordID + vis_state.word);
				if (wordElem !== undefined) word_off(wordElem);
				vis_state.word = 0;
				doc_off(document.getElementById(docID + vis_state.doc));
				var value_new = document.getElementById(docID).value;
				if (!isNaN(value_new) && value_new > 0) {
					value_new = Math.min(D, Math.max(1, value_new));
					doc_on(document.getElementById(docID + value_new));
					vis_state.doc = value_new;
					state_save(true);
					document.getElementById(docID).value = vis_state.doc;
				}
			});

		d3.select("#" + docClear)
			.on("click", function() {
				state_reset();
				state_save(true);
			});


		// Document text border and content div
		var doc_text_div_blue = d3.select("body")
			    .append("div")
			    .attr("id", DocTextBoundaryID)
			    .style("position", "absolute")
			    .style("top", margin.top + 2*margin.pad)
			    .style("background-color", "#5bb0ff")
			    .style("width", mdswidth + margin.pad)
			    .style("height", mdsheight)
			    .style("visibility", "hidden");
		var doc_text_div_white = d3.select("body")
			    .append("div")
			    .attr("id", DocTextID)
			    .style("position", "absolute")
			    .style("top", margin.top + 3*margin.pad)
			    .style("left", "18px")
			    .style("background-color", "white")
			    .style("width", mdswidth - 2*margin.pad)
			    .style("height", mdsheight - 3*margin.pad)
			    .style("overflow-y", "scroll")
			    .style("border-radius", "10px")
			    .style("padding", "5px")
			    .style("visibility", "hidden");


		d3.select("#" + docShow)
			.on("click", function() {

				var doc_to_show = document.getElementById(docID + vis_state.doc);
				var doc_to_show_data = doc_to_show.__data__;

				// docs_text_tooltip.transition()
				// .duration(500)
				// .style("visibility", "visible");

				// var dic_text_div = docs_text_tooltip.append("g")
				// // .attr("x", 0)
				// // .attr("y", 0)
				// // docs_text_tooltip.append("div")
				// // dic_text_div.text(doc_to_show_data.doc_texts)
				// // .style("position", "fixed")
				// // .style("left", "2px")
				// .style("padding", "2px")
				// .style("width", 1*mdswidth + "px")
				// .style("height", mdsheight)
				// // .attr("xml:space", "preserve")
				// // .style("letter-spacing", "10")
				// // .style("word-spacing", "0")
				// // .style("kerning", "20")
				// // .style("top", margin.top + 20 + "px")
				// // .style("z-index", "100")
				// .style("background", "white")
				// .style("border-radius", "10px")
				// .style("overflow-y", "scroll")
				// .style("-webkit-scrollbar-track","background: none;")
				// .style("-webkit-scrollbar", "width: 10px;")
				// .style("-webkit-scrollbar", "height: 10px;")
				// .style("-webkit-scrollbar-thumb", "background: #ccc;")
				// .style("-webkit-scrollbar-thumb", "border-radius: 5px;");



				// var doc_text_div_blue = document.createElement("div");

				// var doc_text_div_blue = d3.select("body")
			 //    .append("div")
			 //    .attr("id", DocTextBoundaryID)
			 //    .style("position", "absolute")
			 //    .style("top", margin.top + 2*margin.pad)
			 //    .style("background-color", "#5bb0ff")
			 //    .style("width", mdswidth + margin.pad)
			 //    .style("height", mdsheight)
			 //    .style("visibility", "visible");

			 //    var doc_text_div_white = d3.select("body")
			 //    .append("div")
			 //    .attr("id", DocTextID)
			 //    .style("position", "absolute")
			 //    .style("top", margin.top + 3*margin.pad)
			 //    .style("left", "18px")
			 //    .style("background-color", "white")
			 //    .text(doc_to_show_data.doc_texts)
			 //    .style("white-space", "pre-wrap")
			 //    .style("width", mdswidth - 2*margin.pad)
			 //    .style("height", mdsheight - 3*margin.pad)
			 //    .style("overflow-y", "scroll")
			 //    .style("border-radius", "10px")
			 //    .style("padding", "5px")
			 //    .style("visibility", "visible");

			    doc_text_div_blue.style("visibility", "visible");
			    doc_text_div_white.text(doc_to_show_data.doc_texts)
			    .style("white-space", "pre-wrap")
			    .style("overflow-y", "scroll")
			    .style("visibility", "visible");

				// doc_text_div_blue.setAttribute("style", "transformOrigin: 0 0; position: absolute; background-color: blue; width: " + mdswidth +"px; height: " + mdsheight + "px; padding: 5px");


				// document.getElementById(visID).appendChild(doc_text_div_blue);

				
				// var doc_text_div_white = document.createElement("div");
				// doc_text_div_white.setAttribute("style", "background-color: white; border-radius: 15px; overflow-y: scroll");
				// .style("border-radius", "15px")
				// .style("overflow", "scroll");

				// doc_text_div_blue.append(doc_text_div_white);

				// var content = document.createTextNode(doc_to_show_data.doc_texts);
				// content.setAttribute("xml:space", "preserve");
				// doc_text_div_white.appendChild(content);

				// doc_text_div_white.addEventListener('scroll', function(evt) {
				//   label.node().setAttribute('y', 10 + this.scrollTop);
				// }, false)

				// doc_text_div_white.addEventListener('scroll');


				// docs_text_tooltip.append("text")
				// .text(doc_to_show_data.doc_texts)
				// .attr("font-family", "georgia")
				// .attr("font-size", "12")
				// // .attr("letter-spacing", "-10")
				// .style("white-space", "pre-wrap");

				
		});

		d3.select("#" + topicID)
			.on("keyup", function() {
				// remove doc selection if it exists (from a saved URL)
				var docElem = document.getElementById(docID + vis_state.doc);
				if (docElem !== undefined) doc_off(docElem);
				vis_state.doc = 0;
				// remove word selection if it exists (from a saved URL)
				var wordElem = document.getElementById(wordID + vis_state.word);
				if (wordElem !== undefined) word_off(wordElem);
				vis_state.word = 0;
				topic_off(document.getElementById(topicID + vis_state.topic));
				var value_new = document.getElementById(topicID).value;
				if (!isNaN(value_new) && value_new > 0) {
					value_new = Math.min(T, Math.max(1, value_new));
					topic_on(document.getElementById(topicID + value_new));
					vis_state.topic = value_new;
					state_save(true);
					document.getElementById(topicID).value = vis_state.topic;
				}
			});

		d3.select("#" + topicClear)
			.on("click", function() {
				state_reset();
				state_save(true);
			});

		d3.select("#" + wordID)
			.on("keyup", function() {
				// remove doc selection if it exists (from a saved URL)
				var docElem = document.getElementById(docID + vis_state.doc);
				if (docElem !== undefined) doc_off(docElem);
				vis_state.doc = 0;
				// remove topic selection if it exists (from a saved URL)
				var topicElem = document.getElementById(topicID + vis_state.topic);
				if (topicElem !== undefined) topic_off(topicElem);
				vis_state.topic = 0;
				word_off(document.getElementById(wordID + vis_state.word));
				var value_new = document.getElementById(wordID).value;
				if (value_new !== 0) {
					// value_new = Math.min(wordMdsData[W], Math.max(1, value_new));
					word_on(document.getElementById(wordID + value_new));
					vis_state.word = value_new;
					state_save(true);
					document.getElementById(wordID).value = vis_state.word;
				}
			});

		d3.select("#" + wordClear)
			.on("click", function() {
				state_reset();
				state_save(true);
			});



		// create linear scaling to pixels (and add some padding on outer region of scatterplot)
		var doc_xrange = d3.extent(docMdsData, function(d) {
			return d.x;
		}); //d3.extent returns min and max of an array
		var doc_xdiff = doc_xrange[1] - doc_xrange[0],
			doc_xpad = 0.05;
		var doc_yrange = d3.extent(docMdsData, function(d) {
			return d.y;
		});
		var doc_ydiff = doc_yrange[1] - doc_yrange[0],
			doc_ypad = 0.05;

		if (doc_xdiff > doc_ydiff) {
			// range: mds width - rMax
			var doc_xScale = d3.scaleLinear()
					.range([rMax, mdswidth - rMax])
					.domain([doc_xrange[0] - doc_xpad * doc_xdiff, doc_xrange[1] + doc_xpad * doc_xdiff]);

			var doc_yScale = d3.scaleLinear()
					.range([mdsheight - rMax, rMax])
					.domain([doc_yrange[0] - 0.5*(doc_xdiff - doc_ydiff) - doc_ypad*doc_xdiff, doc_yrange[1] + 0.5*(doc_xdiff - doc_ydiff) + doc_ypad*doc_xdiff]);
		} else {
			var doc_xScale = d3.scaleLinear()
					.range([rMax, mdswidth - rMax])
					.domain([doc_xrange[0] - 0.5*(doc_ydiff - doc_xdiff) - doc_xpad*doc_ydiff, doc_xrange[1] + 0.5*(doc_ydiff - doc_xdiff) + doc_xpad*doc_ydiff]);

			var doc_yScale = d3.scaleLinear()
					.range([mdsheight - rMax, rMax])
					.domain([doc_yrange[0] - doc_ypad * doc_ydiff, doc_yrange[1] + doc_ypad * doc_ydiff]);
		}

		// create linear scaling to pixels (and add some padding on outer region of scatterplot)
		var topic_xrange = d3.extent(topicMdsData, function(d) {
			return d.x;
		}); //d3.extent returns min and max of an array
		var topic_xdiff = topic_xrange[1] - topic_xrange[0],
			topic_xpad = 0.05;
		var topic_yrange = d3.extent(topicMdsData, function(d) {
			return d.y;
		});
		var topic_ydiff = topic_yrange[1] - topic_yrange[0],
			topic_ypad = 0.05;

		if (topic_xdiff > topic_ydiff) {
			var topic_xScale = d3.scaleLinear()
					.range([rMax, mdswidth - rMax])
					.domain([topic_xrange[0] - topic_xpad * topic_xdiff, topic_xrange[1] + topic_xpad * topic_xdiff]);

			var topic_yScale = d3.scaleLinear()
					.range([mdsheight - rMax, rMax])
					.domain([topic_yrange[0] - 0.5*(topic_xdiff - topic_ydiff) - topic_ypad*topic_xdiff, topic_yrange[1] + 0.5*(topic_xdiff - topic_ydiff) + topic_ypad*topic_xdiff]);
		} else {
			var topic_xScale = d3.scaleLinear()
					.range([rMax, mdswidth - rMax])
					.domain([topic_xrange[0] - 0.5*(topic_ydiff - topic_xdiff) - topic_xpad*topic_ydiff, topic_xrange[1] + 0.5*(topic_ydiff - topic_xdiff) + topic_xpad*topic_ydiff]);

			var topic_yScale = d3.scaleLinear()
					.range([mdsheight - rMax, rMax])
					.domain([topic_yrange[0] - topic_ypad * topic_ydiff, topic_yrange[1] + topic_ypad * topic_ydiff]);
		}

		// create linear scaling to pixels (and add some padding on outer region of scatterplot)
		var word_xrange = d3.extent(wordMdsData, function(d) {
			return d.x;
		}); //d3.extent returns min and max of an array
		var word_xdiff = word_xrange[1] - word_xrange[0],
			word_xpad = 0.05;
		var word_yrange = d3.extent(wordMdsData, function(d) {
			return d.y;
		});
		var word_ydiff = word_yrange[1] - word_yrange[0],
			word_ypad = 0.05;

		if (word_xdiff > word_ydiff) {
			var word_xScale = d3.scaleLinear()
					.range([rMax, mdswidth - rMax])
					.domain([word_xrange[0] - word_xpad * word_xdiff, word_xrange[1] + word_xpad * word_xdiff]);

			var word_yScale = d3.scaleLinear()
					.range([mdsheight - rMax, rMax])
					.domain([word_yrange[0] - 0.5*(word_xdiff - word_ydiff) - word_ypad*word_xdiff, word_yrange[1] + 0.5*(word_xdiff - word_ydiff) + word_ypad*word_xdiff]);
		} else {
			var word_xScale = d3.scaleLinear()
					.range([rMax, mdswidth - rMax])
					.domain([word_xrange[0] - 0.5*(word_ydiff - word_xdiff) - word_xpad*word_ydiff, word_xrange[1] + 0.5*(word_ydiff - word_xdiff) + word_xpad*word_ydiff]);

			var word_yScale = d3.scaleLinear()
					.range([mdsheight - rMax, rMax])
					.domain([word_yrange[0] - word_ypad * word_ydiff, word_yrange[1] + word_ypad * word_ydiff]);
		}

		// Create new svg element (that will contain everything):
		var svg = d3.select(to_select).append("svg")
				.attr("width", (3 * mdswidth) + (2 * margin.left) + (3 * margin.pad))
				.attr("height", mdsheight);

		// Add a group for the doc plot
		var doc_plot = svg.append("g")
				.attr("id", docPanelID)
				.attr("class", "docpoints");


		// Create line element b/w doc and topic plot
		var doc_topic_partition = svg.append("line")
				.style("border", "1px solid black")
       			.attr("x1", 1*mdswidth + margin.pad + (margin.left/2))
       			.attr("x2", 1*mdswidth + margin.pad + (margin.left/2))
       			.attr("y1", 0)
       			.attr("y2", mdsheight)
       			.attr("stroke", "black");

		// Create a group for the topic plot
		var topic_plot = svg.append("g")
				.attr("id", topicPanelID)
				.attr("class", "topicpoints")
				.attr("transform", "translate(" + (1*mdswidth + margin.pad + margin.left) + ", 0)");

		// Create line element b/w doc and topic plot
		var word_topic_partition = svg.append("line")
       			.attr("x1", 2*mdswidth + 2*margin.pad + 3*(margin.left/2))
       			.attr("x2", 2*mdswidth + 2*margin.pad + 3*(margin.left/2))
       			.attr("y1", 0)
       			.attr("y2", mdsheight)
       			.attr("stroke", "black");

		// Add a group for the word plot
		var word_plot = svg.append("g")
				.attr("id", wordPanelID)
				.attr("class", "wordpoints")
				// .attr("align","right")
				.attr("transform", "translate(" + (2*(mdswidth + margin.pad + margin.left)) + ", 0)");


		// Clicking on the doc_plot should clear the selection
		var doc_rec = doc_plot
			.append("rect")
			.style("fill", "none")
			.attr("pointer-events", "all")
			.attr("height", mdsheight)
			.attr("width", (1*mdswidth + margin.pad))
			// .attr("opacity", 0)
			.on("click", function() {
				state_reset();
				state_save(true);
			})
			.call(d3.zoom()
        		.scaleExtent([1, 10])
        		.on("zoom", zoom));

		// var docs_text_tooltip = doc_plot
		// .append("g")
	    // .style("position", "absolute")
		// .style("fill", "#5bb0ff")
	    // .style("z-index", "100")
	    // .style("overflow-y", "scroll")
	    // .style("background", "#5bb0ff")
	    // .style("visibility", "hidden");

		// Clicking on the topic_plot should clear the selection
		topic_plot
			.append("rect")
			.attr("height", mdsheight)
			.attr("width", (1*mdswidth + margin.pad))
			.style("fill", color1)
			.attr("opacity", 0)
			.on("click", function() {
				state_reset();
				state_save(true);
			});

		// Clicking on the word_plot should clear the selection
		word_plot
			.append("rect")
			.attr("height", mdsheight)
			.attr("width", (1*mdswidth + margin.pad))
			.style("fill", color1)
			.attr("opacity", 0)
			.on("click", function() {
				state_reset();
				state_save(true);
			});        


		// bind mdsData to the points in the doc panel and draw circles
		var docpoints = doc_plot.selectAll("circle")
			.data(docMdsData)
			.enter()
			.append("circle")
			.attr("class", "docdot")
			.style("opacity", function(d) {
				return ((d.Freq/10)*0.2);
			})
			.style("fill", color1)
			.attr("r", function(d) {
                // return (rScaleMargin(+d.Freq));
                return (Math.sqrt((d.Freq/(100*Math.PI))*mdswidth*mdsheight*circle_prop/Math.PI));
            })
			.attr("cx", function(d) {
				return (doc_xScale(+d.x));
			})
			.attr("cy", function(d) {
				return (doc_yScale(+d.y));
			})
			.attr("stroke", "black")
			.attr("id", function(d) {
				return (docID + d.docs);
			})
			.attr("transform", transform(d3.zoomIdentity))
			.text(function(d) {
				return d.docs;
			})
			.on("mouseover", function(d) {
				docs_tooltip.text(d.docs); 
				docs_tooltip.style("visibility", "visible");
				var old_doc = docID + vis_state.doc;
				if (vis_state.topic > 0){
					doc_topic_on()
				}
				if (vis_state.doc > 0 && old_doc!= this.id) {
					doc_off(document.getElementById(old_doc));
				}
				doc_on(this);
			})
			.on("click", function(d) {
				// prevent click event defined on the div container from firing
				// http://bl.ocks.org/jasondavies/3186840
				// d3.event.stopPropagation();

				var old_doc = docID + vis_state.doc;
				if (vis_state.doc > 0 && old_doc != this.id) {
					doc_off(document.getElementById(old_doc));
				}
				// make sure doc input box value and fragment reflects clicked selection
				document.getElementById(docID).value = vis_state.doc = d.docs;
				state_save(true);
				doc_on(this);

			})
			.on("mousemove", function(){
				docs_tooltip.style("top", (d3.event.pageY-10)+"px").style("left",(d3.event.pageX+10)+"px");
			})
			.on("mouseout", function(d) {
				docs_tooltip.style("visibility", "hidden");
				if (vis_state.doc != d.docs) doc_off(this);
				if (vis_state.doc > 0) doc_on(document.getElementById(docID + vis_state.doc));
			});;

		var docs_tooltip = d3.select("body")
		    .append("div")
		    .style("position", "absolute")
		    .style("z-index", "10")
		    .style("visibility", "hidden")
		    .attr("stroke", "black")
	    	.text("docs_tooltip");


		function zoom() {
		  docpoints.attr("transform", transform(d3.event.transform));
		}

		function transform(t) {
		  return function(d) {
		    return "translate(" + t.apply([+d.x, +d.y]) + ")";
		  };
		}

		// bind mdsData to the points in the topic panel:
		var topicpoints = topic_plot.selectAll("topicpoints")
				.data(topicMdsData)
				.enter();

		// text to indicate topic
		topicpoints.append("text")
			.attr("class", "topic_txt")
			.attr("x", function(d) {
				return (topic_xScale(+d.x));
			})
			.attr("y", function(d) {
				return (topic_yScale(+d.y) + 4);
			})
			.attr("stroke", "black")
			.attr("opacity", 1)
			.style("text-anchor", "middle")
			.style("font-size", "11px")
			.style("fontWeight", 100)
			.text(function(d) {
				return d.topics;
			});

		// draw circles
		topicpoints.append("circle")
			.attr("class", "topicdot")
			.style("opacity", function(d) {
				return ((d.Freq/10)*0.2);
			})
			.style("fill", color1)
			.attr("r", function(d) {
                // return (rScaleMargin(+d.Freq));
                return (Math.sqrt((d.Freq/(100*Math.PI))*mdswidth*mdsheight*circle_prop/Math.PI));
            })
			.attr("cx", function(d) {
				return (topic_xScale(+d.x));
			})
			.attr("cy", function(d) {
				return (topic_yScale(+d.y));
			})
			.attr("stroke", "black")
			.attr("id", function(d) {
				return (topicID + d.topics);
			})
			.on("mouseover", function(d) {
				var old_topic = topicID + vis_state.topic;
				// actually not greater than zero
				if (vis_state.topic > 0 && old_topic!= this.id) {
					topic_off(document.getElementById(old_topic));
				}
				topic_on(this);
			})
			.on("click", function(d) {
				// prevent click event defined on the div container from firing
				// http://bl.ocks.org/jasondavies/3186840
				d3.event.stopPropagation();
				var old_topic = topicID + vis_state.topic;
				if (vis_state.topic > 0 && old_topic != this.id) {
					topic_off(document.getElementById(old_topic));
				}
				// make sure topic input box value and fragment reflects clicked selection
				document.getElementById(topicID).value = vis_state.topic = d.topics;
				state_save(true);
				topic_on(this);
			})
			.on("mouseout", function(d) {
				if (vis_state.topic != d.topics) topic_off(this);
				if (vis_state.topic > 0) topic_on(document.getElementById(topicID + vis_state.topic));
			});

		// bind mdsData to the points in the word panel:
		var wordpoints = word_plot.selectAll("wordpoints")
				.data(wordMdsData)
				.enter();

		var words_tooltip = d3.select("body")
		    .append("div")
		    .style("position", "absolute")
		    .style("z-index", "10")
		    .style("visibility", "hidden")
		    .attr("stroke", "black")
	    	.text("words_tooltip");

		// draw circles
		wordpoints.append("circle")
			.attr("class", "worddot")
			.style("opacity", function(d) {
				return ((d.Freq/10)*0.2);
			})
			.style("fill", color1)
			.attr("r", function(d) {
                // return (rScaleMargin(+d.Freq));
                if (d.Freq < 0.1)
                	return 0;
                else
                	return (Math.sqrt((d.Freq/(100*Math.PI))*mdswidth*mdsheight*circle_prop/Math.PI));
            })
			.attr("cx", function(d) {
				return (word_xScale(+d.x));
			})
			.attr("cy", function(d) {
				return (word_yScale(+d.y));
			})
			.attr("stroke", "black")
			.attr("id", function(d) {
				return (wordID + d.vocab);
			})
			.text(function(d) {
				return d.vocab;
			})
			.on("mouseover", function(d) {
				words_tooltip.text(d.vocab); 
				words_tooltip.style("visibility", "visible");
				var old_word = wordID + vis_state.word;
				if (typeof vis_state.word === 'string' && old_word!= this.id) {
					word_off(document.getElementById(old_word));
				}
				word_on(this);
			})
			.on("click", function(d) {
				// prevent click event defined on the div container from firing
				// http://bl.ocks.org/jasondavies/3186840
				// d3.event.stopPropagation();
				var old_word = wordID + vis_state.word;
				if (typeof vis_state.word === 'string' && old_word!= this.id) {
					word_off(document.getElementById(old_word));
				}
				// make sure word input box value and fragment reflects clicked selection
				document.getElementById(wordID).value = vis_state.word = d.vocab;
				state_save(true);
				word_on(this);
			})
			.on("mousemove", function(){
				return words_tooltip.style("top", (d3.event.pageY-10)+"px").style("left",(d3.event.pageX+10)+"px");
			})
			.on("mouseout", function(d) {
				words_tooltip.style("visibility", "hidden");
				if (vis_state.word != d.vocab) word_off(this);
				if (typeof vis_state.word === 'string') word_on(document.getElementById(wordID + vis_state.word));
			});


		// dynamically create the doc/topic/word input forms at the top of the page
		function init_forms(docID, topicID, wordID) {

			// create container div for inputs:
			var inputDiv = document.createElement("div");
			inputDiv.setAttribute("id", topID);
			inputDiv.setAttribute("style", "width: 1210px"); // to match the width of the main svg element
			document.getElementById(visID).appendChild(inputDiv);

			// doc input container:
			var docDiv = document.createElement("div");
			docDiv.setAttribute("style", "padding: 5px; background-color: #e8e8e8; display: inline-block; width: " + mdswidth + "px; height: " + margin.top + "px; float: left");
			inputDiv.appendChild(docDiv);

			var docLabel = document.createElement("label");
			docLabel.setAttribute("for", docID);
			docLabel.setAttribute("style", "font-family: sans-serif; font-size: 14px");
			docLabel.innerHTML = "Document: <span id='" + docID + "-value'></span>";
			docDiv.appendChild(docLabel);

			var docInput = document.createElement("input");
			docInput.setAttribute("style", "width: 50px");
			docInput.type = "text";
			// docInput.min = "0";
			// docInput.max = D; // assumes the data has already been read in
			// docInput.step = "1";
			docInput.value = "0"; // a value of 0 indicates no topic is selected
			docInput.id = docID;
			docDiv.appendChild(docInput);

			var show = document.createElement("button");
			show.setAttribute("id", docShow);
			show.setAttribute("style", "margin-left: 5px");
			show.innerHTML = "Show Document";
			docDiv.appendChild(show);

			var clear = document.createElement("button");
			clear.setAttribute("id", docClear);
			clear.setAttribute("style", "margin-left: 5px");
			clear.innerHTML = "Clear Document";
			docDiv.appendChild(clear);

			// topic input container:
			var topicDiv = document.createElement("div");
			topicDiv.setAttribute("style", "padding: 5px; background-color: #e8e8e8; display: inline-block; width: " + mdswidth + "px; height: " + margin.top + "px; margin-left: 5px");
			inputDiv.appendChild(topicDiv);

			var topicLabel = document.createElement("label");
			topicLabel.setAttribute("for", topicID);
			topicLabel.setAttribute("style", "font-family: sans-serif; font-size: 14px");
			topicLabel.innerHTML = "Topic: <span id='" + topicID + "-value'></span>";
			topicDiv.appendChild(topicLabel);

			var topicInput = document.createElement("input");
			topicInput.setAttribute("style", "width: 50px");
			topicInput.type = "text";
			// topicInput.min = "0";
			// topicInput.max = T; // assumes the data has already been read in
			// topicInput.step = "1";
			topicInput.value = "0"; // a value of 0 indicates no topic is selected
			topicInput.id =  topicID;
			topicDiv.appendChild(topicInput);

			var clear = document.createElement("button");
			clear.setAttribute("id", topicClear);
			clear.setAttribute("style", "margin-left: 5px");
			clear.innerHTML = "Clear Topic";
			topicDiv.appendChild(clear);

			// word input container:
			var wordDiv = document.createElement("div");
			wordDiv.setAttribute("style", "padding: 5px; background-color: #e8e8e8; display: inline-block; width: " + mdswidth + "px; height: " + margin.top + "px; margin-left: 5px");
			inputDiv.appendChild(wordDiv);

			var wordLabel = document.createElement("label");
			wordLabel.setAttribute("for", wordID);
			wordLabel.setAttribute("style", "font-family: sans-serif; font-size: 14px");
			wordLabel.innerHTML = "Word: <span id='" + wordID + "-value'></span>";
			wordDiv.appendChild(wordLabel);

			var wordInput = document.createElement("input");
			wordInput.setAttribute("style", "width: 50px");
			wordInput.type = "text";
			// wordInput.min = "0";
			// wordInput.max = W; // assumes the data has already been read in
			// wordInput.step = "1";
			wordInput.value = "0"; // a value of 0 indicates no word is selected
			wordInput.id = wordID;
			wordDiv.appendChild(wordInput);

			var clear = document.createElement("button");
			clear.setAttribute("id", wordClear);
			clear.setAttribute("style", "margin-left: 5px");
			clear.innerHTML = "Clear Word";
			wordDiv.appendChild(clear);

		}

		var rScaleCond = d3.scaleSqrt()
                    .domain([0, 1]).range([0, rMax]);

		//////////////////////////////////////////////////////////////////////////////

		// function to update topic/word plot when a doc is selected
		// the circle argument should be the appropriate circle element
		function doc_on(circle) {
			if (circle == null) return null;

			// grab data bound to this element
			var d = circle.__data__;
			var doc_index = d.index;

			// change opacity and fill of the selected circle
			circle.style.opacity = highlight_opacity;
			circle.style.fill = color2;


			// word interactions

			// grab the word-plot data for this doc only:
			var dat1 = doc_word_info.filter(function(d) {
				return d.Doc == doc_index;
			});

			var w = dat1.length; // number of words for this doc

			// freq depicted using color intensity rather than radius  (T = total vocab)
			var word_radius = [];
			for (var i = 0; i < W; ++i) {
				word_radius[i] = 0;
			}
			for (i = 0; i < w; i++) {
				word_radius[dat1[i].Word-1] = dat1[i].Freq;
			}

			// Change color of bubbles according to the doc's distribution over words
			d3.selectAll(to_select + " .worddot")
				.data(word_radius)
				.transition()
				.attr("r", function(d) {
					//return (rScaleCond(d));
					return (Math.sqrt((d/(100*Math.PI))*mdswidth*mdsheight*circle_prop/Math.PI));
				});

			// re-bind mdsData so we can handle multiple selection
			d3.selectAll(to_select + " .worddot")
				.data(wordMdsData);


			// topic interactions

			var dat2 = doc_topic_info.filter(function(d) {
				return d.Doc == doc_index;
			});

			var t = dat2.length; // number of topics for this doc

			var topic_radius = [];
			for (var i = 0; i < T; ++i) {
				topic_radius[i] = 0;
			}
			for (i = 0; i < t; i++) {
				topic_radius[dat2[i].Topic-1] = dat2[i].Freq;
			}

			// Change color of bubbles according to the doc's distribution over topics
			d3.selectAll(to_select + " .topicdot")
				.data(topic_radius)
				.transition()
				.attr("r", function(d) {
					return (Math.sqrt((d/(100*Math.PI))*mdswidth*mdsheight*circle_prop/Math.PI));
				});

			// re-bind mdsData so we can handle multiple selection
			d3.selectAll(to_select + " .topicdot")
				.data(topicMdsData);
		}

		// function to update doc/word plot when a topic is selected
		// the circle argument should be the appropriate circle element
		function topic_on(circle) {
			if (circle == null) return null;

			// grab data bound to this element
			var d = circle.__data__;
			var topic_index = d.index;

			// change opacity and fill of the selected circle
			circle.style.opacity = highlight_opacity;
			circle.style.fill = color2;

			// doc interactions

			// grab the doc-plot data for this topic only:
			var dat1 = topic_doc_info.filter(function(d) {
				return d.Topic == topic_index;
			});

			var dd = dat1.length; // number of docs for this topic

			// freq depicted using color intensity rather than radius  (T = total vocab)
			var doc_radius = [];
			for (var i = 0; i < D; ++i) {
				doc_radius[i] = 0;
			}
			for (i = 0; i < dd; i++) {
				doc_radius[dat1[i].Doc-1] = dat1[i].Freq;
			}

			d3.selectAll(to_select + " .docdot")
				.data(doc_radius)
				.transition()
				.attr("r", function(d) {
					return (Math.sqrt((d/(100*Math.PI))*mdswidth*mdsheight*circle_prop/Math.PI));
				});

			// re-bind mdsData so we can handle multiple selection
			d3.selectAll(to_select + " .docdot")
				.data(docMdsData);


			// word interactions

			var dat2 = topic_word_info.filter(function(d) {
				return d.Topic == topic_index;
			});

			var w = dat2.length; // number of words for this topic

			var word_radius = [];
			for (var i = 0; i < T; ++i) {
				word_radius[i] = 0;
			}
			for (i = 0; i < w; i++) {
				word_radius[dat2[i].Word-1] = dat2[i].Freq;
			}

			// Change color of bubbles according to the topic's distribution over word
			d3.selectAll(to_select + " .worddot")
				.data(word_radius)
				.transition()
				.attr("r", function(d) {
					return (Math.sqrt((d/(100*Math.PI))*mdswidth*mdsheight*circle_prop/Math.PI));
				});

			// re-bind mdsData so we can handle multiple selection
			d3.selectAll(to_select + " .worddot")
				.data(wordMdsData);

		}


		// function to update doc/topic plot when a word is selected
		// the circle argument should be the appropriate circle element
		function word_on(circle) {
			if (circle == null) return null;

			// grab data bound to this element
			var d = circle.__data__;
			var word_index = d.index;

			// change opacity and fill of the selected circle
			circle.style.opacity = highlight_opacity;
			circle.style.fill = color2;

			// doc interactions

			// grab the doc-plot data for this word only:
			var dat1 = word_doc_info.filter(function(d) {
				return d.Word == word_index;
			});
			

			var dd = dat1.length; // number of docs for this word

			// freq depicted using color intensity rather than radius  (T = total vocab)
			var doc_radius = [];
			for (var i = 0; i < D; ++i) {
				doc_radius[i] = 0;
			}
			for (i = 0; i < dd; i++) {
				// add 1 to escape circle disappearence case when Freq is 0
				doc_radius[dat1[i].Doc-1] = dat1[i].Freq;
			}


			// Change color of bubbles according to the word's distribution over docs

			d3.selectAll(to_select + " .docdot")
				.data(doc_radius)
				.transition()
				.attr("r", function(d) {
					return (Math.sqrt((d/(100*Math.PI))*mdswidth*mdsheight*circle_prop/Math.PI));
				});

			// re-bind mdsData so we can handle multiple selection
			d3.selectAll(to_select + " .docdot")
				.data(docMdsData);

			// topic interactions
			// grab the topic-plot data for this word only:
			var dat2 = word_topic_info.filter(function(d) {
				return d.Word == word_index;
			});

			var t = dat2.length; // number of topics for this word

			var topic_radius = [];
			for (var i = 0; i < T; ++i) {
				topic_radius[i] = 0;
			}
			for (i = 0; i < t; i++) {
				topic_radius[dat2[i].Topic-1] = dat2[i].Freq;
			}

			// Change color of bubbles according to the doc's distribution over topics]
			d3.selectAll(to_select + " .topicdot")
				.data(topic_radius)
				.transition()
				.attr("r", function(d) {
					return (Math.sqrt((d/(100*Math.PI))*mdswidth*mdsheight*circle_prop/Math.PI));
				});

			// re-bind mdsData so we can handle multiple selection
			d3.selectAll(to_select + " .topicdot")
				.data(topicMdsData);

		}

		function doc_off(circle) {
			if (circle == null) return circle;
			// go back to original opacity/fill
			circle.style.opacity = base_opacity;
			circle.style.fill = color1;

			d3.selectAll(to_select + " .topicdot")
				.data(topicMdsData)
				.transition()
				.attr("r", function(d) {
					//return (rScaleMargin(+d.Freq));
					return (Math.sqrt((d.Freq/(100*Math.PI))*mdswidth*mdsheight*circle_prop/Math.PI));
				});

			d3.selectAll(to_select + " .worddot")
				.data(wordMdsData)
				.transition()
				.attr("r", function(d) {
					//return (rScaleMargin(+d.Freq));
					return (Math.sqrt((d.Freq/(100*Math.PI))*mdswidth*mdsheight*circle_prop/Math.PI));
				});

			// // Change sizes of topic numbers:
			// d3.selectAll(to_select + " .txt")
			//     .transition()
			//     .style("font-size", "11px");

		}

		function topic_off(circle) {
			if (circle == null) return circle;
			// go back to original opacity/fill
			circle.style.opacity = base_opacity;
			circle.style.fill = color1;

			d3.selectAll(to_select + " .docdot")
				.data(docMdsData)
				.transition()
				.attr("r", function(d) {
					//return (rScaleMargin(+d.Freq));
					return (Math.sqrt((d.Freq/(100*Math.PI))*mdswidth*mdsheight*circle_prop/Math.PI));
				});

			d3.selectAll(to_select + " .worddot")
				.data(wordMdsData)
				.transition()
				.attr("r", function(d) {
					//return (rScaleMargin(+d.Freq));
					return (Math.sqrt((d.Freq/(100*Math.PI))*mdswidth*mdsheight*circle_prop/Math.PI));
				});

			// // Change sizes of topic numbers:
			// d3.selectAll(to_select + " .txt")
			//     .transition()
			//     .style("font-size", "11px");

		}

		function word_off(circle) {
			if (circle == null) return circle;
			// go back to original opacity/fill
			circle.style.opacity = base_opacity;
			circle.style.fill = color1;

			d3.selectAll(to_select + " .docdot")
				.data(docMdsData)
				.transition()
				.attr("r", function(d) {
					//return (rScaleMargin(+d.Freq));
					return (Math.sqrt((d.Freq/(100*Math.PI))*mdswidth*mdsheight*circle_prop/Math.PI));
				});

			d3.selectAll(to_select + " .topicdot")
				.data(topicMdsData)
				.transition()
				.attr("r", function(d) {
					//return (rScaleMargin(+d.Freq));
					return (Math.sqrt((d.Freq/(100*Math.PI))*mdswidth*mdsheight*circle_prop/Math.PI));
				});

			// // Change sizes of topic numbers:
			// d3.selectAll(to_select + " .txt")
			//     .transition()
			//     .style("font-size", "11px");

		}

		// serialize the visualization state using fragment identifiers -- http://en.wikipedia.org/wiki/Fragment_identifier
		// location.hash holds the address information

		var params = location.hash.split("&");
		if (params.length > 1) {
			vis_state.doc = params[0].split("=")[1];
			vis_state.topic = params[1].split("=")[1];
			vis_state.word = params[2].split("=")[1];

			// Idea: write a function to parse the URL string
			// only accept values in [0,1] for lambda, {0, 1, ..., K} for topics (any string is OK for term)
			// Allow for subsets of the three to be entered:
			// (1) topic only (lambda = 1 term = "")
			// (2) lambda only (topic = 0 term = "") visually the same but upon hovering a topic, the effect of lambda will be seen
			// (3) term only (topic = 0 lambda = 1) only fires when the term is among the R most salient
			// (4) topic + lambda (term = "")
			// (5) topic + term (lambda = 1)
			// (6) lambda + term (topic = 0) visually lambda doesn't make a difference unless a topic is hovered
			// (7) topic + lambda + term

			// Short-term: assume format of "#topic=k&lambda=l&term=s" where k, l, and s are strings (b/c they're from a URL)

			// Force t (doc identifier) to be an integer between 0 and D:
			vis_state.doc = Math.round(Math.min(D, Math.max(0, vis_state.doc)));
			// Force t (topic identifier) to be an integer between 0 and T:
			vis_state.topic = Math.round(Math.min(T, Math.max(0, vis_state.topic)));
			// Force w (word identifier) to be an integer between 0 and W:
			vis_state.word = Math.round(Math.min(W, Math.max(0, vis_state.word)));

			// select the doc
			if (!isNaN(vis_state.doc)) {
				document.getElementById(docID).value = vis_state.doc;
				if (vis_state.doc > 0) {
					doc_on(document.getElementById(docID + vis_state.doc));
				}
			}

			// select the topic
			if (!isNaN(vis_state.topic)) {
				document.getElementById(topicID).value = vis_state.topic;
				if (vis_state.topic > 0) {
					topic_on(document.getElementById(topicID + vis_state.topic));
				}
			}

			// select the word
			if (!isNaN(vis_state.word)) {
				document.getElementById(wordID).value = vis_state.word;
				if (vis_state.word > 0) {
					word_on(document.getElementById(wordID + vis_state.word));
				}
			}
		}


		function state_url() {
			return location.origin + location.pathname + "#doc=" + vis_state.doc +
				"&topic=" + vis_state.topic + "&word=" + vis_state.word;
		}

		function state_save(replace) {
			if (replace)
				history.replaceState(vis_state, "Query", state_url());
			else
				history.pushState(vis_state, "Query", state_url());
		}

		function state_reset() {
			if (vis_state.doc > 0) {
				doc_off(document.getElementById(docID + vis_state.doc));
			}
			if (vis_state.topic > 0) {
				topic_off(document.getElementById(topicID + vis_state.topic));
			}
			if (typeof vis_state.word === 'string') {
				word_off(document.getElementById(wordID + vis_state.word));
			}

			document.getElementById(docID).value = vis_state.doc = 0;
			document.getElementById(topicID).value = vis_state.topic = 0;
			document.getElementById(wordID).value = vis_state.word = 0;

			var doc_text_display = document.getElementById(DocTextID)
			var doc_text_boundary = document.getElementById(DocTextBoundaryID)

			if (doc_text_display.style.visibility !== "hidden") {
				doc_text_display.style.visibility = "hidden";
				doc_text_boundary.style.visibility = "hidden";
        	}

			// if (docs_text_tooltip.visibility !== 'hidden') {
			// 	docs_text_tooltip.transition()
			// 	.duration(500)
			// 	.style("visibility", "hidden");
			// }

			state_save(true);
		}

	}

	if (typeof data_or_file_name === 'string')
		d3.json(data_or_file_name, function(error, data) {visualize(data);});
	else
		visualize(data_or_file_name);

	// var current_clicked = {
	//     what: "nothing",
	//     element: undefined
	// },

	//debugger;

};