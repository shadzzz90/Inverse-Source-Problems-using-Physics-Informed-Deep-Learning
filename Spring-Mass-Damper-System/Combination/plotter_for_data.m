clc
clear all 
fileno = 0;
duffingNL3simdata = readtable('.\Trial5\duffingNL4_sim_data.csv','NumHeaderLines',1);
t = table2array(duffingNL3simdata(:,1));
input_vec = table2array(duffingNL3simdata(:,4));
disp =table2array(duffingNL3simdata(:,2));
vel = table2array(duffingNL3simdata(:,3));


subplot(2,2,1)
plot(t,input_vec,'-b','MarkerSize',2,'LineWidth',1.5)
xlabel('$t$','FontSize', 13,'Interpreter','latex');
ylabel('$f(t)$','FontSize', 13,'Interpreter','latex');
xlim([0 50]);
grid on
set(gca,'GridLineStyle',':','GridColor', 'k','GridAlpha',1 )
title('$f(t)$ vs $t$','FontSize', 13,'Interpreter','latex')


subplot(2,2,2)
plot(t,disp,'-r','MarkerSize',2,'LineWidth',1.5)
xlabel('$t$','FontSize', 13,'Interpreter','latex');
ylabel('$x$','FontSize', 13,'Interpreter','latex');
xlim([0 50]);
ylim([-4 4]);
grid on
set(gca,'GridLineStyle',':','GridColor', 'k','GridAlpha',1 )
title('$x$ vs $t$','FontSize', 13,'Interpreter','latex')


subplot(2,2,3)
plot(t,vel,'-r','MarkerSize',2,'LineWidth',1.5)
xlabel('$t$','FontSize', 13,'Interpreter','latex');
ylabel('$\dot{x}$','FontSize',13,'Interpreter','latex');
xlim([0 50]);
ylim([-4 4]);
grid on
set(gca,'GridLineStyle',':','GridColor', 'k','GridAlpha',1 )
title('$\dot{x}$ vs $t$','FontSize', 13,'Interpreter','latex')

subplot(2,2,4)
plot(disp,vel,'xb','MarkerSize',1,'LineWidth',1.5)
xlabel('$x$','FontSize', 13,'Interpreter','latex');
ylabel('$\dot{x}$','FontSize',13,'Interpreter','latex');
xlim([-4 4]);
ylim([-4 4]);
grid on
set(gca,'GridLineStyle',':','GridColor', 'k','GridAlpha',1 )
title('phase plot','FontSize', 13,'Interpreter','latex')

saveas(gcf,sprintf('%02d_plot.eps',fileno),'epsc')