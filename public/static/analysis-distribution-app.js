const app = {
    data() {
        return {
            isVisible: false,
            isVisible2: true,
            stock: '',
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

            this.alertError('success', 'Request sent to the server');
            fetch('/analysis/distribution', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    ticker: this.stock,
                    type: 'distribution'
                })
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
                const layout = {
                    template: data.layout.template,
                    uirevision: data.layout.uirevision,
                    showlegend: data.layout.showlegend,
                    title: data.layout.title,
                    annotations: data.layout.annotations,
                    xaxis: data.layout.xaxis,
                    xaxis2: data.layout.xaxis2,
                    xaxis3: data.layout.xaxis3,
                    xaxis4: data.layout.xaxis4,
                    xaxis5: data.layout.xaxis5,
                    yaxis: data.layout.yaxis,
                    yaxis2: data.layout.yaxis2,
                    yaxis3: data.layout.yaxis3,
                    yaxis4: data.layout.yaxis4,
                    yaxis5: data.layout.yaxis5,
                };

                Plotly.newPlot('stockDistributionGraphData', plotlyData, layout);
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