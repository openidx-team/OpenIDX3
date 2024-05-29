const app = {
    data() {
        return {
            isVisible: false,
            isVisible2: true,
            stock: '',
            model: 'LSTM',
            alertMessage: '',
            alertType: ''
        };
    },
    methods: {
        toggleMenu() {
            this.isVisible = !this.isVisible;
        },
        submitBackend() {
            this.stock = this.stock.toUpperCase().replace('.JK', '');

            if (this.stock.trim() === '') {
                this.alertError('error', 'Please enter a stock ticker');
                return;
            }

            if (this.model.trim() === '') {
                this.alertError('error', 'Please select a model');
                return;
            }

            this.alertError('success', 'Request sent to the server');
            fetch('/analysis/prediction', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    ticker: this.stock,
                    type: 'prediction',
                    model: this.model,
                }),
            })
            .then(response => response.json())
            .then(data => {
                if (!data) {
                    this.alertError('error', 'No data returned from the server');
                    return;
                }
                else if (data.error) {
                    this.alertError('error', data.error);
                    return;
                }
                const plotlyData = data.data;
                let layout = {};

                if (this.model === 'ARIMA') {
                    layout = {
                        template: data.layout.template,
                        uirevision: data.layout.uirevision,
                        showlegend: data.layout.showlegend,
                        title: data.layout.title,
                        xaxis: data.layout.xaxis,
                        yaxis: data.layout.yaxis,
                    };
                }
                else if (this.model === 'LSTM') {
                    layout = {
                        template: data.layout.template,
                        uirevision: data.layout.uirevision,
                        showlegend: data.layout.showlegend,
                        title: data.layout.title,
                        xaxis: data.layout.xaxis,
                        yaxis: data.layout.yaxis,
                        shapes: data.layout.shapes,
                    };
                }

                Plotly.newPlot('stockPredictionnGraphData', plotlyData, layout);
            })
            .catch((error) => {
                this.alertError('Error', error);
            });
        },
        alertError(type, message) {
            this.alertType = type;
            this.alertMessage = message;
        },
        dismissAlert() {
            this.alertType = '';
            this.alertMessage = '';
        },
        resetAlert() {
            this.dismissAlert(); 
            this.isVisible2 = true; 
        }
    },
    mounted() {
        setInterval(() => {
            this.isVisible2 = false;
            this.resetAlert(); 
        }, 5000);
    },
    delimiters: ['[[',']]']
}

Vue.createApp(app).mount('#app')