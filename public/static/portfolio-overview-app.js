const app = Vue.createApp({
    data() {
        return {
            isVisible: false,
            isVisible2: true
        };
    },
    methods: {
        toggleMenu() {
            this.isVisible = !this.isVisible;
        }
    },
    mounted() {
        setTimeout(() => {
            this.isVisible2 = false;
        }, 5000);
    },
    delimiters: ['[[',']]']
})

app.mount('#app')

