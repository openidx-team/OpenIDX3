const app = {
    data() {
        return {
            isVisible: false,
            isVisible2: true,
            stock: '',
            alertMessage: '',
            alertType: '',
            sentimentData: [], 
            sentimentScore: 0,
            showSentiment: false
        };
    },
    computed: {
        sentimentScoreMap() {
            return {
                positive: 1,
                neutral: 0,
                negative: -1
            };
        }
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
            fetch('/analysis/sentiment', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    ticker: this.stock,
                    type: 'sentiment',
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

                if (data.news) {
                    this.sentimentData = Object.values(data.news);
                    this.showSentiment = true;
                } else {
                    this.alertError('error', 'No sentiment data found');
                }
                
            })
            .catch((error) => {
                this.alertError('Error', error);
            });
        },
        calculateSentimentScore(data) {
            let score = 0;
            for (let i = 0; i < data.length; i++) {
                score += this.sentimentScoreMap[data[i].sentiment];
            }
            score = score / data.length;
            return score;
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
    watch: {
        sentimentData: {
            handler: function (data) {
                this.sentimentScore = this.calculateSentimentScore(data);
            },
            deep: true
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
