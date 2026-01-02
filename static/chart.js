// function drawChart(models, scores) {
//     var data = [{
//         x: models,
//         y: scores,
//         type: 'bar'
//     }];

//     var layout = {
//         title: 'Model Comparison',
//         xaxis: { title: 'Models' },
//         yaxis: { title: 'Score' }
//     };

//     Plotly.newPlot('chart', data, layout);
// }



document.addEventListener("DOMContentLoaded", function () {
    const chartDiv = document.getElementById("chart");

    const models = JSON.parse(chartDiv.dataset.models);
    const scores = JSON.parse(chartDiv.dataset.scores);

    var data = [{
        x: models,
        y: scores,
        type: 'bar'
    }];

    var layout = {
        title: 'Model Comparison',
        xaxis: { title: 'Models' },
        yaxis: { title: 'Score' }
    };

    Plotly.newPlot('chart', data, layout);
});
