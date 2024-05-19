const app = {
    data() {
        return {
            isVisible: false,
            isVisible2: true,
            newStock: '',
            stocks: [],
            jsonData: [],
            alertMessage: '',
            alertType: ''
        }
    },
    methods: {
        toggleMenu() {
            this.isVisible = !this.isVisible
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
            fetch('/analysis/fundamental', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ 
                    ticker: this.stocks, 
                    type: 'fundamental'
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
                this.jsonData = data;
                this.ticker = '';
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
Vue.createApp(app).mount('#app')