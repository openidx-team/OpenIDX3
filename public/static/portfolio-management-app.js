const app = Vue.createApp({
    data() {
        return {
            isVisible: false,
            isVisible2: true,
            showEditModal: false,
            showDeleteModal: false,
            editedTicker: '',
            editedShares: 0,
            editedBuyPrice: 0,
            editedBuyDate: ''
        };
    },
    methods: {
        toggleMenu() {
            this.isVisible = !this.isVisible;
        },
        toggleEditModal(ticker, shares, buy_price, buy_date) {
            this.editedTicker = ticker;
            this.editedShares = shares;
            this.editedBuyPrice = buy_price;
            this.editedBuyDate = buy_date;
            this.showEditModal = !this.showEditModal;
        },
        toggleDeleteModal(ticker, shares, buy_price, buy_date) {
            this.editedTicker = ticker;
            this.editedShares = shares;
            this.editedBuyPrice = buy_price;
            this.editedBuyDate = buy_date;           
            this.showDeleteModal = !this.showDeleteModal;
        }
    },
    mounted() {
        setTimeout(() => {
            this.isVisible2 = false;
        }, 5000);
    },
    delimiters: ['[[', ']]']
});

app.mount('#app');