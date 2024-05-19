const app = {
    data() {
        return {
            isVisible: false,
            isVisible2: true,
            newStock: '',
            numPortfolios: 10000,
            riskFreeRate: 0.065,
            stocks: [],
            maxSharpeWeights: [],
            minVolatilityWeights: [],
            alertMessage: '',
            alertType: ''
        }
    },
    methods: {
        toggleMenu() {
            this.isVisible = !this.isVisible;
        },
        addStock() {
            if (this.newStock.trim() !== '') { 
                this.newStock = this.newStock.toUpperCase();
                this.stocks.push(this.newStock);
                this.newStock = ''; 
            }
        },
        removeStock(index) {
            this.stocks.splice(index, 1);
        },
        setPortfolio() {
            this.stocks = [];
            this.stocks = ['portfolio'];
            this.submitBackend();
            this.stocks = [];
        },
        submitBackend() {
            if (this.stocks.length === 0) {
                this.alertError('error', 'Please add a stock to be analyzed');
                return;
            }
            if (this.stocks.length === 1) {
                if (this.stocks[0] !== 'portfolio') {
                    this.alertError('error', 'Please add more than one stock to be analyzed');
                    return;
                }
            }
            if (this.riskFreeRate === '') {
                this.alertError('error', 'Please enter a risk-free rate');
                return;
            }
            if (this.numPortfolios === '') {
                this.alertError('error', 'Please enter the number of portfolios to generate');
                return;
            }
            if (this.riskFreeRate < 0) {
                this.alertError('error', 'Risk-free rate must be between 0 and 1');
                return;
            }
            if (this.riskFreeRate > 1) {
                this.alertError('error', 'Risk-free rate must be between 0 and 1');
                return;
            }
            if (this.numPortfolios < 1) {
                this.alertError('error', 'Number of portfolios must be between 1 and 10000');
                return;
            }
            if (this.numPortfolios > 10000) {
                this.alertError('error', 'Number of portfolios must be between 1 and 10000');
                return;
            }

            this.alertError('success', 'Request sent to the server');
            fetch('/analysis/optimization', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ 
                    ticker: this.stocks, 
                    type: 'optimization',
                    numPortfolios: this.numPortfolios,
                    riskFreeRate: this.riskFreeRate
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
                    title: data.layout.title.text, 
                    xaxis: data.layout.xaxis,
                    yaxis: data.layout.yaxis,
                    legend: {
                        y: 0.5,
                        yref: 'paper',
                        font: { family: 'Arial, sans-serif', size: 20, color: 'grey' },
                    }
                };

                this.maxSharpeWeights = data.maxSharpeWeights;
                this.minVolatilityWeights = data.minVolWeights;

                Plotly.newPlot('stock_optimization_graph', plotlyData, layout);
            })
            .catch(error => {
                this.alertError('error', error);
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
    computed: {
        filteredStocks() {
            return this.stocks.filter(stock => stock !== 'portfolio');
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

Vue.createApp(app).mount('#app');
