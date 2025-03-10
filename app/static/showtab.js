function showTab(tabName) {
    // Hide all tabs
    const tabs = document.querySelectorAll('.tab-content');
    tabs.forEach(tab => tab.style.display = 'none');

    // Show the selected tab
    document.getElementById(tabName).style.display = 'block';

    // Remove active class from all tabs
    const tabButtons = document.querySelectorAll('.tab');
    tabButtons.forEach(tab => tab.classList.remove('active'));

    // Add active class to the clicked tab
    const activeTab = document.querySelector(`.tab[onclick="showTab('${tabName}')"]`);
    if (activeTab) activeTab.classList.add('active');

    // Store the selected tab in localStorage
    localStorage.setItem('lastTab', tabName);
}


//const lastTab = 'search';
//showTab(lastTab);